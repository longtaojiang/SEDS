from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import copy
import torch
import pickle as pkl
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn, KL
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip, RGB_Encoder
from modules.module_fusionencoder import MLP_feature_fusion, Gloss_Fusion_Transformer
import  torch.nn.functional as F
from modules.module_clip import CLIP, CLIP_vision, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from modules.modeling_signbert import init_sign_model

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.clip_rgb = None
        self.cross = None
        self.distributed = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None,distributed=False, type_vocab_size=2, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()
        for key, val in clip_state_dict.items():
            if key.find('visual') > -1:
                new_key = "clip_rgb." + key
                if new_key not in state_dict:
                    state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)
        model.distributed=distributed
        ## ===> Initialization trick [HARD CODE]

        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        self._stage_one = True
        self._stage_two = False
        self.signbert_have = task_config.signbert
        self.fusion_type = task_config.fusion_type
        self.freeze_exfusion = task_config.freeze_exfusion
        self.dual_mix = task_config.dual_mix
        self.mix_design = task_config.mix_design
        self.rgb_pose_kl = task_config.rgb_pose_kl
        self.kl_pose_loss = task_config.kl_pose_loss
        self.kl_rgb_loss = task_config.kl_rgb_loss
        self.kl_logit = task_config.kl_logit
        self.rgb_pose_match = task_config.rgb_pose_match
        self.rgb_pose_match_loss = task_config.rgb_pose_match_loss

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32
        vision_layers=self.task_config.visual_num_hidden_layers
        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t rgb_dim: {}".format(task_config.rgb_dim))
        show_log(task_config, "\t pose_dim: {}".format(task_config.pose_dim))
        show_log(task_config, "\t fusion_type: {}".format(task_config.fusion_type))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,feature_len=task_config.feature_len, input_size=task_config.pose_dim,
            linear_patch=self.linear_patch
        ).float()
        self.clip_rgb = CLIP_vision(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,feature_len=task_config.feature_len, input_size=task_config.rgb_dim,
            linear_patch=self.linear_patch
        ).float()
        self.aug_choose=task_config.aug_choose
        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        convert_weights(self.clip_rgb)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        
        if self.fusion_type == "mlp":
            self.fusion = MLP_feature_fusion(input_channel=task_config.hidden_dim*2, output_channel=task_config.hidden_dim)
        elif self.fusion_type == "gloss_atten":
            self.fusion = Gloss_Fusion_Transformer(hidden_size=task_config.hidden_dim)

        if self.signbert_have:
            self.signbert = init_sign_model(args=task_config)

        self.loss_fct = CrossEn()

        if self.rgb_pose_kl:
            self.loss_kl = KL(self.kl_logit)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, right_batch, left_batch, body_batch, input_ids_aug=None, attention_mask_aug=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        input_ids_aug = input_ids_aug.view(-1, input_ids.shape[-1])

        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        attention_mask_aug=attention_mask_aug.view(-1, attention_mask.shape[-1])

        sequence_hidden, text_mask, visual_hidden_pose, video_mask, visual_hidden_rgb, sequence_hidden_aug, text_mask_aug= self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         right_batch, left_batch, body_batch, shaped=True, input_ids_aug=input_ids_aug, attention_mask_aug=attention_mask_aug)

        if self.training:
            loss = 0.
            loss_pose_kl = 0.
            loss_rgb_kl = 0.

            if self.sim_header == "Filip":
                I2T_sim_fusion, T2I_sim_fusion, I2T_sim_pose, T2I_sim_pose, I2T_sim_rgb, T2I_sim_rgb, loss_pose_kl, loss_rgb_kl, P2R_sim, R2P_sim = self.get_similarity_logits(sequence_hidden, visual_hidden_pose, visual_hidden_rgb, text_mask,
                                                                     video_mask,
                                                                     shaped=True, loose_type=self.loose_type,sequence_hidden_aug=sequence_hidden_aug,text_mask_aug=text_mask_aug)

                # sim_loss=sim_loss1
                sim_loss1=self.loss_fct(I2T_sim_fusion)*self.dual_mix+ self.loss_fct(I2T_sim_fusion.T)*(1-self.dual_mix)
                sim_loss2=self.loss_fct(T2I_sim_fusion.T)*self.dual_mix+self.loss_fct(T2I_sim_fusion)*(1-self.dual_mix)
                sim_loss = (sim_loss1 + sim_loss2) / 2

                sim_loss1_pose=self.loss_fct(I2T_sim_pose)*self.dual_mix+ self.loss_fct(I2T_sim_pose.T)*(1-self.dual_mix)
                sim_loss2_pose=self.loss_fct(T2I_sim_pose.T)*self.dual_mix+self.loss_fct(T2I_sim_pose)*(1-self.dual_mix)
                sim_loss_pose = (sim_loss1_pose + sim_loss2_pose) / 2

                sim_loss1_rgb=self.loss_fct(I2T_sim_rgb)*self.dual_mix+ self.loss_fct(I2T_sim_rgb.T)*(1-self.dual_mix)
                sim_loss2_rgb=self.loss_fct(T2I_sim_rgb.T)*self.dual_mix+self.loss_fct(T2I_sim_rgb)*(1-self.dual_mix)
                sim_loss_rgb = (sim_loss1_rgb + sim_loss2_rgb) / 2

                if self.rgb_pose_kl:
                    loss_pose_kl = self.kl_pose_loss * loss_pose_kl
                    loss_rgb_kl = self.kl_rgb_loss * loss_rgb_kl
                
                if self.rgb_pose_match:
                    sim_loss1_r2p=self.loss_fct(P2R_sim)*self.dual_mix+ self.loss_fct(P2R_sim.T)*(1-self.dual_mix)
                    sim_loss2_r2p=self.loss_fct(R2P_sim.T)*self.dual_mix+self.loss_fct(R2P_sim)*(1-self.dual_mix)
                    sim_loss_r2p = (sim_loss1_r2p + sim_loss2_r2p) / 2
                    sim_loss_r2p = sim_loss_r2p * self.rgb_pose_match_loss
                else:
                    sim_loss_r2p = 0.
                
                if self.freeze_exfusion:
                    loss += sim_loss
                else:
                    loss += sim_loss + sim_loss_pose + sim_loss_rgb + loss_pose_kl + loss_rgb_kl + sim_loss_r2p

            return loss, sim_loss, sim_loss_pose, sim_loss_rgb, loss_pose_kl, loss_rgb_kl, sim_loss_r2p
        else:
            return None

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False,get_hidden=True):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        if self.sim_header=='Filip'and get_hidden==True:
            text_mask,sequence_hidden = self.clip.encode_text(input_ids,return_hidden=True)  # B,512

            sequence_hidden=sequence_hidden.float()
            return text_mask, sequence_hidden

        return sequence_hidden

    def get_sign_output(self, right_batch, left_batch, body_batch):
        
        clips_start = body_batch['clips_start']
        clip_mask = body_batch['mask']
        rgb_feature = body_batch['rgb']
        batch_num, feature_len = clips_start.size()
        slide_windows = self.task_config.slide_windows
        
        pose_all = {}
        pose_all['right'] = right_batch['pose']
        pose_all['left'] = left_batch['pose']
        pose_all['body'] = body_batch['pose']

        del right_batch, left_batch, body_batch
        torch.cuda.empty_cache()

        pose_all = self.signbert.gcn_emb(pose_all)
        batch_num, seq_length, feat_dim = pose_all['feat'].size()
        pose_final_new = torch.zeros((batch_num, feature_len, slide_windows, feat_dim)).to(device=pose_all['feat'].device, dtype=pose_all['feat'].dtype)

        for i in range(batch_num):
            for j in range(feature_len):
                if clips_start[i, j] != -1:
                    assert clip_mask[i, j+1] == 0
                    pose_final_new[i, j, :, :] = pose_all['feat'][i, clips_start[i,j]:clips_start[i,j]+slide_windows, :]
                else:
                    assert clip_mask[i, j+1] == 1

        del pose_all
        torch.cuda.empty_cache()
        

        rgb_final = rgb_feature

        pose_final_new = pose_final_new.reshape(batch_num*feature_len, slide_windows, feat_dim)
        pose_final_new = self.signbert.sign_conv(pose_final_new)
        pose_final_new = pose_final_new.reshape(batch_num, feature_len, slide_windows, feat_dim)
        pose_final_new = torch.mean(pose_final_new, dim=-2)
        
        pose_final_new = pose_final_new.permute(0, 2, 1).unsqueeze(-1)
        
        return rgb_final, pose_final_new, clip_mask

    def get_visual_output(self, right_batch, left_batch, body_batch, shaped=True, get_hidden=True):
        
        video_rgb, video_pose, video_mask = self.get_sign_output(right_batch, left_batch, body_batch)

        bs_pair = video_mask.size(0)
        video_frame=1

        if self.sim_header == 'Filip' and get_hidden==True:
            _, visual_hidden_pose = self.clip.encode_image(video_pose, return_hidden=True, video_mask=video_mask, video_frame=video_frame)
            visual_hidden_pose=visual_hidden_pose.float()
            _, visual_hidden_rgb = self.clip_rgb.encode_image(video_rgb, return_hidden=True, video_mask=video_mask, video_frame=video_frame)
            visual_hidden_rgb=visual_hidden_rgb.float()

            visual_hidden_pose = visual_hidden_pose.view(bs_pair, -1, visual_hidden_pose.size(-1))
            visual_hidden_rgb = visual_hidden_rgb.view(bs_pair, -1, visual_hidden_rgb.size(-1))

        else:
            ValueError("no such sim_header!!!")

        return video_mask, visual_hidden_pose, visual_hidden_rgb

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, right_batch, left_batch, body_batch, shaped=False, input_ids_aug=None, attention_mask_aug=None):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        
        text_mask, sequence_hidden = self.get_sequence_output(input_ids,  token_type_ids, attention_mask, shaped=False)
        text_mask_aug, sequence_hidden_aug = self.get_sequence_output(input_ids_aug,  token_type_ids, attention_mask_aug, shaped=False)

        video_mask, visual_hidden_pose, visual_hidden_rgb = self.get_visual_output(right_batch, left_batch, body_batch, shaped=True)

        return sequence_hidden, text_mask, visual_hidden_pose, video_mask, visual_hidden_rgb, sequence_hidden_aug, text_mask_aug
    
    def kl_double_modal_compute(self, batch_size, t_len, v_len, video_mask, attention_mask, text_mask_aug, i2t_sim_pose, i2t_sim_aug_pose, i2t_sim_rgb, i2t_sim_aug_rgb):

        with torch.no_grad():
            torch.set_printoptions(profile="full")
            kl_eye_mask = torch.eye(batch_size, device=video_mask.device).bool().unsqueeze(2).unsqueeze(3)
            kl_video_mask = video_mask.unsqueeze(2).repeat(1,1,t_len)
            kl_text_mask = attention_mask.unsqueeze(1).repeat(1,v_len,1)
            kl_textaug_mask = text_mask_aug.unsqueeze(1).repeat(1,v_len,1)

            teancher_pose_vt = torch.masked_select(i2t_sim_pose * torch.softmax(i2t_sim_pose/0.07, dim=3), kl_eye_mask).reshape(batch_size, v_len, t_len)
            teancher_pose_vt[~kl_text_mask] = -1000000.0
            teancher_pose_vt = torch.masked_select(teancher_pose_vt, kl_video_mask).reshape(-1, t_len)
            _, pose_vt_idx = torch.topk(teancher_pose_vt, 3, dim=-1)
            pose_vt_idx_mask = torch.ones_like(teancher_pose_vt).bool()
            pose_vt_idx_mask.scatter_(1, pose_vt_idx, False)
            teancher_pose_vt[pose_vt_idx_mask] = -1000000.0

            teancher_pose_tv = torch.masked_select(i2t_sim_aug_pose * torch.softmax(i2t_sim_aug_pose/0.07, dim=2), kl_eye_mask).reshape(batch_size, v_len, t_len)
            teancher_pose_tv[~kl_video_mask] = -1000000.0
            teancher_pose_tv = torch.masked_select(teancher_pose_tv.permute(0, 2, 1), kl_textaug_mask.permute(0, 2, 1)).reshape(-1, v_len)
            _, pose_tv_idx = torch.topk(teancher_pose_tv, 5, dim=-1)
            pose_tv_idx_mask = torch.ones_like(teancher_pose_tv).bool()
            pose_tv_idx_mask.scatter_(1, pose_tv_idx, False)
            teancher_pose_tv[pose_tv_idx_mask] = -1000000.0

            teancher_rgb_vt = torch.masked_select(i2t_sim_rgb * torch.softmax(i2t_sim_rgb/0.07, dim=3), kl_eye_mask).reshape(batch_size, v_len, t_len)
            teancher_rgb_vt[~kl_text_mask] = -1000000.0
            teancher_rgb_vt = torch.masked_select(teancher_rgb_vt, kl_video_mask).reshape(-1, t_len)
            _, rgb_vt_idx = torch.topk(teancher_rgb_vt, 3, dim=-1)
            rgb_vt_idx_mask = torch.ones_like(teancher_rgb_vt).bool()
            rgb_vt_idx_mask.scatter_(1, rgb_vt_idx, False)
            teancher_rgb_vt[rgb_vt_idx_mask] = -1000000.0

            teancher_rgb_tv = torch.masked_select(i2t_sim_aug_rgb * torch.softmax(i2t_sim_aug_rgb/0.07, dim=2), kl_eye_mask).reshape(batch_size, v_len, t_len)
            teancher_rgb_tv[~kl_video_mask] = -1000000.0
            teancher_rgb_tv = torch.masked_select(teancher_rgb_tv.permute(0, 2, 1), kl_textaug_mask.permute(0, 2, 1)).reshape(-1, v_len)
            _, rgb_tv_idx = torch.topk(teancher_rgb_tv, 5, dim=-1)
            rgb_tv_idx_mask = torch.ones_like(teancher_rgb_tv).bool()
            rgb_tv_idx_mask.scatter_(1, rgb_tv_idx, False)
            teancher_rgb_tv[rgb_tv_idx_mask] = -1000000.0

        stu_pose_vt = torch.masked_select(i2t_sim_pose * torch.softmax(i2t_sim_pose/0.07, dim=3), kl_eye_mask).reshape(batch_size, v_len, t_len)
        stu_pose_vt[~kl_text_mask] = -1000000.0
        stu_pose_vt = torch.masked_select(stu_pose_vt, kl_video_mask).reshape(-1, t_len)
        stu_pose_vt[rgb_vt_idx_mask] = -1000000.0
        stu_pose_tv = torch.masked_select(i2t_sim_aug_pose * torch.softmax(i2t_sim_aug_pose/0.07, dim=2), kl_eye_mask).reshape(batch_size, v_len, t_len)
        stu_pose_tv[~kl_video_mask] = -1000000.0
        stu_pose_tv = torch.masked_select(stu_pose_tv.permute(0, 2, 1), kl_textaug_mask.permute(0, 2, 1)).reshape(-1, v_len)
        stu_pose_tv[rgb_tv_idx_mask] = -1000000.0
        loss_pose_kl = 0.5 * (self.loss_kl(stu_pose_vt, teancher_rgb_vt) + self.loss_kl(stu_pose_tv, teancher_rgb_tv))

        stu_rgb_vt = torch.masked_select(i2t_sim_rgb * torch.softmax(i2t_sim_rgb/0.07, dim=3), kl_eye_mask).reshape(batch_size, v_len, t_len)
        stu_rgb_vt[~kl_text_mask] = -1000000.0
        stu_rgb_vt = torch.masked_select(stu_rgb_vt, kl_video_mask).reshape(-1, t_len)
        stu_rgb_vt[pose_vt_idx_mask] = -1000000.0
        stu_rgb_tv = torch.masked_select(i2t_sim_aug_rgb * torch.softmax(i2t_sim_aug_rgb/0.07, dim=2), kl_eye_mask).reshape(batch_size, v_len, t_len)
        stu_rgb_tv[~kl_video_mask] = -1000000.0
        stu_rgb_tv = torch.masked_select(stu_rgb_tv.permute(0, 2, 1), kl_textaug_mask.permute(0, 2, 1)).reshape(-1, v_len)
        stu_rgb_tv[pose_tv_idx_mask] = -1000000.0
        loss_rgb_kl = 0.5 * (self.loss_kl(stu_rgb_vt, teancher_pose_vt) + self.loss_kl(stu_rgb_tv, teancher_pose_tv))

        return loss_pose_kl, loss_rgb_kl
    
    def flip_similarity_softmax(self, sequence_output, visual_hidden_pose, visual_hidden_rgb, attention_mask, video_mask, sim_header="meanP",pad_type=1,sequence_hidden_aug=None,text_mask_aug=None):
        
        visual_hidden_fusion = self.fusion(visual_hidden_pose, visual_hidden_rgb, video_mask)

        if self.training and self.distributed:
            visual_hidden_pose = allgather(visual_hidden_pose, self.task_config)
            visual_hidden_rgb = allgather(visual_hidden_rgb, self.task_config)
            visual_hidden_fusion = allgather(visual_hidden_fusion, self.task_config)

            video_mask = allgather(video_mask, self.task_config)

            sequence_output = allgather(sequence_output, self.task_config)
            sequence_hidden_aug = allgather(sequence_hidden_aug, self.task_config)
            attention_mask = allgather(attention_mask, self.task_config)
            text_mask_aug=allgather(text_mask_aug, self.task_config)
            
            torch.distributed.barrier()

        video_mask = (video_mask == 0)
        attention_mask = (attention_mask==1)
        text_mask_aug = (text_mask_aug==1)

        visual_hidden_pose = visual_hidden_pose / visual_hidden_pose.norm(dim=-1, keepdim=True)
        visual_hidden_pose = visual_hidden_pose.squeeze(1)

        visual_hidden_rgb = visual_hidden_rgb / visual_hidden_rgb.norm(dim=-1, keepdim=True)
        visual_hidden_rgb = visual_hidden_rgb.squeeze(1)

        visual_hidden_fusion = visual_hidden_fusion / visual_hidden_fusion.norm(dim=-1, keepdim=True)
        visual_hidden_fusion = visual_hidden_fusion.squeeze(1)

        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        sequence_output = sequence_output.squeeze(1)

        batch_size, v_len=visual_hidden_pose.shape[0],visual_hidden_pose.shape[1]
        batch_size_t, t_len=sequence_output.shape[0],sequence_output.shape[1]

        sequence_hidden_aug = sequence_hidden_aug / sequence_hidden_aug.norm(dim=-1, keepdim=True)
        sequence_hidden_aug = sequence_hidden_aug.squeeze(1)

        logit_scale = self.clip.logit_scale.exp()
        
        #pose
        i2t_sim_pose=torch.einsum("ais, bjs->abij", [visual_hidden_pose, sequence_output])

        # i2t_sim_pose_cpu = i2t_sim_pose.cpu()
        # i2t_sim_pose_np = i2t_sim_pose_cpu.numpy()
        # with open(os.path.join(self.task_config.output_dir, 'i2t_sim_pose.pkl'), 'wb') as f:
        #     pkl.dump(i2t_sim_pose_np, f)

        i2t_sim_aug_pose=torch.einsum("ais, bjs->abij", [visual_hidden_pose, sequence_hidden_aug])

        after_softmax_i2t_pose = torch.nansum(i2t_sim_pose * torch.softmax(i2t_sim_pose/0.07, dim=3), dim=3)
        video_mask_extend=video_mask.unsqueeze(1).repeat(1,batch_size_t,1)
        after_softmax_i2t_pose[~video_mask_extend]=0
        I2T_sim_pose = logit_scale*torch.nansum(after_softmax_i2t_pose, dim=-1)/torch.sum(video_mask_extend,dim=-1)

        after_softmax_t2i_pose = torch.nansum(i2t_sim_aug_pose * torch.softmax(i2t_sim_aug_pose/0.07, dim=2), dim=2)
        text_mask_extend2=text_mask_aug.unsqueeze(0).repeat(batch_size,1,1)
        after_softmax_t2i_pose[~text_mask_extend2]=0
        T2I_sim_pose = logit_scale*torch.nansum(after_softmax_t2i_pose*text_mask_extend2, dim=-1)/torch.sum(text_mask_extend2,dim=-1)

        #rgb
        i2t_sim_rgb=torch.einsum("ais,bjs->abij", [visual_hidden_rgb, sequence_output])

        # i2t_sim_rgb_cpu = i2t_sim_rgb.cpu()
        # i2t_sim_rgb_np = i2t_sim_rgb_cpu.numpy()
        # with open(os.path.join(self.task_config.output_dir, 'i2t_sim_rgb.pkl'), 'wb') as f:
        #     pkl.dump(i2t_sim_rgb_np, f)
        
        i2t_sim_aug_rgb=torch.einsum("ais,bjs->abij", [visual_hidden_rgb, sequence_hidden_aug])

        after_softmax_i2t_rgb = torch.nansum(i2t_sim_rgb * torch.softmax(i2t_sim_rgb/0.07, dim=3), dim=3)
        video_mask_extend=video_mask.unsqueeze(1).repeat(1,batch_size_t,1)
        after_softmax_i2t_rgb[~video_mask_extend]=0
        I2T_sim_rgb = logit_scale*torch.nansum(after_softmax_i2t_rgb, dim=-1)/torch.sum(video_mask_extend,dim=-1)

        after_softmax_t2i_rgb = torch.nansum(i2t_sim_aug_rgb * torch.softmax(i2t_sim_aug_rgb/0.07, dim=2), dim=2)
        text_mask_extend2=text_mask_aug.unsqueeze(0).repeat(batch_size,1,1)
        after_softmax_t2i_rgb[~text_mask_extend2]=0
        T2I_sim_rgb = logit_scale*torch.nansum(after_softmax_t2i_rgb*text_mask_extend2, dim=-1)/torch.sum(text_mask_extend2,dim=-1)

        #fusion
        i2t_sim_fusion=torch.einsum("ais,bjs->abij", [visual_hidden_fusion, sequence_output])

        # i2t_sim_fusion_cpu = i2t_sim_fusion.cpu()
        # i2t_sim_fusion_np = i2t_sim_fusion_cpu.numpy()
        # with open(os.path.join(self.task_config.output_dir, 'i2t_sim_fusion.pkl'), 'wb') as f:
        #     pkl.dump(i2t_sim_fusion_np, f)

        i2t_sim_aug_fusion=torch.einsum("ais,bjs->abij", [visual_hidden_fusion, sequence_hidden_aug])

        after_softmax_i2t_fusion = torch.nansum(i2t_sim_fusion * torch.softmax(i2t_sim_fusion/0.07, dim=3), dim=3)
        video_mask_extend=video_mask.unsqueeze(1).repeat(1,batch_size_t,1)
        after_softmax_i2t_fusion[~video_mask_extend]=0
        I2T_sim_fusion = logit_scale*torch.nansum(after_softmax_i2t_fusion, dim=-1)/torch.sum(video_mask_extend,dim=-1)

        after_softmax_t2i_fusion = torch.nansum(i2t_sim_aug_fusion * torch.softmax(i2t_sim_aug_fusion/0.07, dim=2), dim=2)
        text_mask_extend2=text_mask_aug.unsqueeze(0).repeat(batch_size,1,1)
        after_softmax_t2i_fusion[~text_mask_extend2]=0
        T2I_sim_fusion = logit_scale*torch.nansum(after_softmax_t2i_fusion*text_mask_extend2, dim=-1)/torch.sum(text_mask_extend2,dim=-1)

        if self.training and self.rgb_pose_kl:
            loss_pose_kl, loss_rgb_kl = self.kl_double_modal_compute(batch_size, t_len, v_len, video_mask, attention_mask, text_mask_aug, i2t_sim_pose, i2t_sim_aug_pose, i2t_sim_rgb, i2t_sim_aug_rgb)
        else:
            loss_pose_kl = 0.0
            loss_rgb_kl = 0.0

        if self.training and self.rgb_pose_match:
            #rgb and pose match
            pose2rgb_sim=torch.einsum("ais, bjs->abij", [visual_hidden_pose, visual_hidden_rgb])
            sim_eye_mask = torch.eye(v_len, device = pose2rgb_sim.device).unsqueeze(0).unsqueeze(0)
            assert len(sim_eye_mask.shape) == 4 and sim_eye_mask.shape[0] == 1 and sim_eye_mask.shape[1] == 1
            sim_eye_mask = sim_eye_mask.repeat(batch_size, batch_size, 1, 1)

            after_softmax_pose2rgb = torch.nansum(sim_eye_mask * pose2rgb_sim * torch.softmax(pose2rgb_sim/0.07, dim=3), dim=3)
            video_mask_extend=video_mask.unsqueeze(1).repeat(1, batch_size, 1)
            after_softmax_pose2rgb[~video_mask_extend]=0
            P2R_sim = logit_scale*torch.nansum(after_softmax_pose2rgb, dim=-1)

            after_softmax_rgb2pose = torch.nansum(sim_eye_mask * pose2rgb_sim * torch.softmax(pose2rgb_sim/0.07, dim=2), dim=2)
            video_mask_extend=video_mask.unsqueeze(0).repeat(batch_size, 1, 1)
            after_softmax_rgb2pose[~video_mask_extend]=0
            R2P_sim = logit_scale*torch.nansum(after_softmax_rgb2pose, dim=-1)
        else:
            P2R_sim = None
            R2P_sim = None

        return I2T_sim_fusion, T2I_sim_fusion, I2T_sim_pose, T2I_sim_pose, I2T_sim_rgb, T2I_sim_rgb, loss_pose_kl, loss_rgb_kl, P2R_sim, R2P_sim

    def get_similarity_logits(self, sequence_output, visual_hidden_pose, visual_hidden_rgb, attention_mask, video_mask, shaped=False, loose_type=False,is_train=True,sequence_hidden_aug=None,text_mask_aug=None):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if sequence_hidden_aug==None:
            sequence_hidden_aug=sequence_output
            text_mask_aug=attention_mask

        if video_mask[0][0] == 1:
            ValueError("the video_mask[0][0] == 1")
        
        if self.sim_header=='Filip' and is_train==True:
            
            I2T_sim_fusion, T2I_sim_fusion, I2T_sim_pose, T2I_sim_pose, I2T_sim_rgb, T2I_sim_rgb, loss_pose_kl, loss_rgb_kl, P2R_sim, R2P_sim = self.flip_similarity_softmax(sequence_output, visual_hidden_pose, visual_hidden_rgb, attention_mask, video_mask,
                                                     sim_header=self.sim_header,sequence_hidden_aug=sequence_hidden_aug,text_mask_aug=text_mask_aug)

            return I2T_sim_fusion, T2I_sim_fusion, I2T_sim_pose, T2I_sim_pose, I2T_sim_rgb, T2I_sim_rgb, loss_pose_kl, loss_rgb_kl, P2R_sim, R2P_sim
        
        return None, None, None
    

# with torch.no_grad():
#     torch.set_printoptions(profile="full")
#     kl_eye_mask = torch.eye(batch_size, device=visual_hidden_fusion.device).bool().unsqueeze(2).unsqueeze(3)
#     kl_video_mask = video_mask.unsqueeze(2).repeat(1,1,t_len)
#     kl_text_mask = attention_mask.unsqueeze(1).repeat(1,v_len,1)
#     kl_textaug_mask = text_mask_aug.unsqueeze(1).repeat(1,v_len,1)
#     teancher_pose_vt = torch.masked_select(i2t_sim_pose * torch.softmax(i2t_sim_pose/0.07, dim=3), kl_eye_mask).reshape(batch_size, v_len, t_len)
#     teancher_pose_vt[~kl_text_mask] = -1000000.0
#     teancher_pose_vt = torch.masked_select(teancher_pose_vt, kl_video_mask).reshape(-1, t_len)
#     _, pose_vt_idx = torch.topk(teancher_pose_vt, 3, dim=-1)
#     pose_vt_idx_mask = torch.ones_like(teancher_pose_vt).bool()
#     pose_vt_idx_mask.scatter_(1, pose_vt_idx, False)
#     teancher_pose_vt[pose_vt_idx_mask] = -1000000.0
#     teancher_pose_tv = torch.masked_select(i2t_sim_aug_pose * torch.softmax(i2t_sim_aug_pose/0.07, dim=2), kl_eye_mask).reshape(batch_size, v_len, t_len)
#     teancher_pose_tv[~kl_video_mask] = -1000000.0
#     teancher_pose_tv = torch.masked_select(teancher_pose_tv.permute(0, 2, 1), kl_textaug_mask.permute(0, 2, 1)).reshape(-1, v_len)
#     _, pose_tv_idx = torch.topk(teancher_pose_tv, 5, dim=-1)
#     pose_tv_idx_mask = torch.ones_like(teancher_pose_tv).bool()
#     pose_tv_idx_mask.scatter_(1, pose_tv_idx, False)
#     teancher_pose_tv[pose_tv_idx_mask] = -1000000.0
#     teancher_rgb_vt = torch.masked_select(i2t_sim_rgb * torch.softmax(i2t_sim_rgb/0.07, dim=3), kl_eye_mask).reshape(batch_size, v_len, t_len)
#     teancher_rgb_vt[~kl_text_mask] = -1000000.0
#     teancher_rgb_vt = torch.masked_select(teancher_rgb_vt, kl_video_mask).reshape(-1, t_len)
#     _, rgb_vt_idx = torch.topk(teancher_rgb_vt, 3, dim=-1)
#     rgb_vt_idx_mask = torch.ones_like(teancher_rgb_vt).bool()
#     rgb_vt_idx_mask.scatter_(1, rgb_vt_idx, False)
#     teancher_rgb_vt[rgb_vt_idx_mask] = -1000000.0
#     teancher_rgb_tv = torch.masked_select(i2t_sim_aug_rgb * torch.softmax(i2t_sim_aug_rgb/0.07, dim=2), kl_eye_mask).reshape(batch_size, v_len, t_len)
#     teancher_rgb_tv[~kl_video_mask] = -1000000.0
#     teancher_rgb_tv = torch.masked_select(teancher_rgb_tv.permute(0, 2, 1), kl_textaug_mask.permute(0, 2, 1)).reshape(-1, v_len)
#     _, rgb_tv_idx = torch.topk(teancher_rgb_tv, 5, dim=-1)
#     rgb_tv_idx_mask = torch.ones_like(teancher_rgb_tv).bool()
#     rgb_tv_idx_mask.scatter_(1, rgb_tv_idx, False)
#     teancher_rgb_tv[rgb_tv_idx_mask] = -1000000.0
# stu_pose_vt = torch.masked_select(i2t_sim_pose * torch.softmax(i2t_sim_pose/0.07, dim=3), kl_eye_mask).reshape(batch_size, v_len, t_len)
# stu_pose_vt[~kl_text_mask] = -1000000.0
# stu_pose_vt = torch.masked_select(stu_pose_vt, kl_video_mask).reshape(-1, t_len)
# stu_pose_vt[rgb_vt_idx_mask] = -1000000.0
# stu_pose_tv = torch.masked_select(i2t_sim_aug_pose * torch.softmax(i2t_sim_aug_pose/0.07, dim=2), kl_eye_mask).reshape(batch_size, v_len, t_len)
# stu_pose_tv[~kl_video_mask] = -1000000.0
# stu_pose_tv = torch.masked_select(stu_pose_tv.permute(0, 2, 1), kl_textaug_mask.permute(0, 2, 1)).reshape(-1, v_len)
# stu_pose_tv[rgb_tv_idx_mask] = -1000000.0
# loss_pose_kl = 0.5 * (self.loss_kl(stu_pose_vt, teancher_rgb_vt) + self.loss_kl(stu_pose_tv, teancher_rgb_tv))
# stu_rgb_vt = torch.masked_select(i2t_sim_rgb * torch.softmax(i2t_sim_rgb/0.07, dim=3), kl_eye_mask).reshape(batch_size, v_len, t_len)
# stu_rgb_vt[~kl_text_mask] = -1000000.0
# stu_rgb_vt = torch.masked_select(stu_rgb_vt, kl_video_mask).reshape(-1, t_len)
# stu_rgb_vt[pose_vt_idx_mask] = -1000000.0
# stu_rgb_tv = torch.masked_select(i2t_sim_aug_rgb * torch.softmax(i2t_sim_aug_rgb/0.07, dim=2), kl_eye_mask).reshape(batch_size, v_len, t_len)
# stu_rgb_tv[~kl_video_mask] = -1000000.0
# stu_rgb_tv = torch.masked_select(stu_rgb_tv.permute(0, 2, 1), kl_textaug_mask.permute(0, 2, 1)).reshape(-1, v_len)
# stu_rgb_tv[pose_tv_idx_mask] = -1000000.0
# loss_rgb_kl = 0.5 * (self.loss_kl(stu_rgb_vt, teancher_pose_vt) + self.loss_kl(stu_rgb_tv, teancher_pose_tv))