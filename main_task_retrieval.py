from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from datetime import datetime
import torch
import numpy as np
import random
import pickle as pkl
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam
from modules.modeling_signbert import Sign_Bert
import matplotlib.pyplot as plt
import time

from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT
global logger
from nltk import data
data.path.append(r"/aiarena/gpfs/jlt/nltk_data/")
data.path.append(r"/data/jlt/jlt_signtvr_mm_2024/nltk_data/")


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65535, rlimit[1]))

torch.distributed.init_process_group(backend="nccl")

def get_args(description='CLCL on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train",action='store_true' , help="Whether to run training.")
    parser.add_argument("--do_eval",action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--distributed", default=True, help="Whether to imply ditributed learning, must set as True for correct gathering.")
    parser.add_argument("--debug", action='store_true', help="Whether to debug.")


    ##########  DA  ##########
    parser.add_argument('--choose_agnostic_aware_sum', type=int, default=3, help='feature use type')
    parser.add_argument('--combine_type', type=str, default='sum', help='feature combine type')
    parser.add_argument('--video_path', type=str, default='', help='video path')
    parser.add_argument('--features_path', type=str, default='sign_features/h2s_domain_agnostic', help='feature path')
    parser.add_argument('--features_RGB_path', type=str, default='sign_features/h2s_domain_aware', help='feature path')
    parser.add_argument('--alpha', type=float, default=0.8, help='feature combine weight')
    ##########  DA  ##########

    ##########  CL  ##########
    parser.add_argument("--dual_mix", default=0.5,type=float, help="Mix weight for two similarity matrix")
    parser.add_argument("--mix_design", default='balance',type=str, help="similarity matrix combine type")
    parser.add_argument("--tau", default=0.07,type=float, help="Learning temperature")
    parser.add_argument("--sim_calcu", default='softmax_max', type=str, help="similarity matrix combine type")
    ##########  CL  ##########


    ##########  TA  ##########
    parser.add_argument('--text_aug', default=True, help='whether to use text augmentation')
    parser.add_argument('--text_aug_choosen', type=str, default='random_swap', help='feature path',choices=['synonym_replacement','random_deletion','random_swap','all'])
    parser.add_argument('--aug_choose', type=str, default='t2v')
    ##########  TA  ##########

    ##########  Net Archi  ##########
    parser.add_argument('--coef_lr', type=float, default=1.0, help='coefficient for bert branch.')
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--not_load_visual', default=False, help="Layer NO. of CLIP need to freeze.")
    ##########  Net Archi  ##########

    ##########  Learning paras  ##########
    parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate')
    parser.add_argument('--sign_lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
    parser.add_argument('--stop_epochs', type=int, default=210, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=256, help='batch size eval')
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--init_sign_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    ##########  Learning paras  ##########

    ##########  Token length   ##########
    parser.add_argument('--max_words', type=int, default=32, help='')
    parser.add_argument('--feature_len', type=int ,default=64, help="Whether MIL, has a high priority than use_mil.")
    parser.add_argument('--max_length_frames', type=int ,default=300, help="")
    parser.add_argument('--slide_windows', type=int ,default=16, help="")
    parser.add_argument('--windows_stride', type=int ,default=1, help="")
    parser.add_argument('--original_size', default=512, type=int)
    parser.add_argument('--original_size_w', default=256, type=int)
    parser.add_argument('--original_size_h', default=256, type=int)
    parser.add_argument('--crop_size', default=256, type=int)
    parser.add_argument('--threshold', default=0.0, type=float)
    parser.add_argument('--frames_threshold', default=0.0, type=float)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    parser.add_argument('--interval', default=2, type=int)
    ##########  Token length   ##########
    
    ##########  SignBert paras   ##########
    parser.add_argument("--rgb_pose_match", action='store_true', help="")
    parser.add_argument('--rgb_pose_match_loss', type=float, default=0.4, help='')
    parser.add_argument('--fusion_type', default='mlp', type=str, choices='mlp, gloss_atten')
    parser.add_argument("--rgb_pose_kl", action='store_true', help="")
    parser.add_argument('--kl_pose_loss', type=float, default=0.5, help='')
    parser.add_argument('--kl_rgb_loss', type=float, default=0.5, help='')
    parser.add_argument('--kl_logit', type=float, default=0.01, help='')
    parser.add_argument("--signbert", action='store_true', help="")
    parser.add_argument('--hidden_dim', type=int, default=512, help='')
    parser.add_argument('--rgb_dim', type=int, default=1024, help='')
    parser.add_argument('--pose_dim', type=int, default=1536, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--heads', type=int, default=8, help='')
    parser.add_argument('--d_ff', type=int, default=2048, help='')
    parser.add_argument('--blocks', type=int, default=3, help='')
    parser.add_argument('--in_channels', type=int, default=2, help='')
    parser.add_argument('--layout_encoder', default='stb', type=str, help='')
    parser.add_argument('--strategy', default='spatial', type=str, help='')
    parser.add_argument('--temporal_pad', default=0, type=int, help='the connection of temporal dimension ')
    ##########  SignBert paras   ##########

    parser.add_argument('--data_path', type=str, default='data_h2', help='data pickle file path')
    parser.add_argument('--cross_att_layers', type=int, default=1, help='feature path')
    parser.add_argument('--num_thread_reader', type=int, default=8, help='')
    parser.add_argument('--lr_decay', type=float, default=0.001, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=10, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_frames', type=int, default=128, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')
    parser.add_argument('--gpu_ids', type=list, default=[0,1], help='The GPU used when distributed is off')
    parser.add_argument('--cla_weight_dir', type=str, default='cla_weight', help='Num of pair to output from data loader')


    parser.add_argument("--output_dir", default='new_experiment', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', default=True,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="h2s", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', default=True, help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="Filip",
                        choices=["Filip", "meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str, help="Choose a CLIP version")

    args = parser.parse_args()


    if args.sim_header == "tightTransf":
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.distributed==True:
        world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(args.local_rank)
        args.world_size = world_size
        rank = torch.distributed.get_rank()
        args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device):

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        for key in model_state_dict.keys():
            if key.find("signbert") > -1:
                args.init_sign_model = None
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir,distributed=args.distributed, state_dict=model_state_dict, task_config=args)
    model.to(device)

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module
        
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if ("clip." in n) or ("clip_rgb." in n) ]
    decay_sign_param_tp = [(n, p) for n, p in decay_param_tp if "signbert." in n ]
    decay_noclipsign_param_tp = [(n, p) for n, p in decay_param_tp if ("clip." not in n) and ("clip_rgb." not in n) and ("signbert." not in n)]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if ("clip." in n) or ("clip_rgb." in n) ]
    no_decay_sign_param_tp = [(n, p) for n, p in no_decay_param_tp if "signbert." in n]
    no_decay_noclipsign_param_tp = [(n, p) for n, p in no_decay_param_tp if ("clip." not in n) and ("clip_rgb." not in n) and ("signbert." not in n)]

    weight_decay = 0.001
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_sign_param_tp], 'weight_decay': weight_decay, 'lr': args.sign_lr * coef_lr},
        {'params': [p for n, p in decay_noclipsign_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_sign_param_tp], 'weight_decay': 0.0, 'lr': args.sign_lr * coef_lr},
        {'params': [p for n, p in no_decay_noclipsign_param_tp], 'weight_decay': 0.0},
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.sign_lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    if args.distributed==True:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids).cuda()

    return optimizer, scheduler, model

def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def save_best_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_best_model.bin")
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_best_opt.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': tr_loss,
            }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)

        for key in model_state_dict.keys():
            if key.find("signbert") > -1:
                args.init_sign_model = None

        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    if args.debug==True:
        print("model allocated:")
        print(torch.cuda.memory_allocated()/1024.0/1024)

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = {k:v.to(device) for k,v in batch.items()}

        if args.debug == True:
            print("input allocated:")
            print(torch.cuda.memory_allocated()/1024.0/1024)
        
        sample = batch
        right_batch = {'pose':sample['right_pose']}
        left_batch = {'pose':sample['left_pose']}
        body_batch = {'pose':sample['body_pose'], 'clips_start':sample['body_clips_start'], 'mask':sample['body_mask'], 'rgb':sample['RGB_feature']}
        
        input_ids = sample['pairs_text']
        input_mask = sample['pairs_mask']
        segment_ids = sample['pairs_segment']
        pairs_text_aug = sample['pairs_text_aug']
        pairs_mask_aug = sample['pairs_mask_aug']

        loss, loss_fusion, loss_pose, loss_rgb, kl_pose, kl_rgb, loss_r2p = model(input_ids, segment_ids, input_mask, right_batch, left_batch, body_batch, pairs_text_aug, pairs_mask_aug)

        if args.debug == True:
            print("forward allocated:")
            print(torch.cuda.memory_allocated()/1024.0/1024)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        torch.cuda.empty_cache()
        loss.backward()

        if args.debug == True:
            print("backward allocated:")
            print(torch.cuda.memory_allocated()/1024.0/1024)
        torch.cuda.empty_cache()
        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            else:
                torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch:%d/%s, Step:%d/%d, Loss:%f, Loss_f:%f, Loss_p:%f, Loss_r:%f, r2p:%f, T/S:%.3f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader),
                            float(loss),
                            float(loss_fusion),
                            float(loss_pose),
                            float(loss_rgb),
                            float(loss_r2p),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu_new_mix(model, batch_list_v, batch_list_t, batch_sequence_output_list, batch_visual_output_pose_list, batch_visual_output_rgb_list, is_train, hybird=False, alpha=0.5,dual_mix=0.5):

    with torch.no_grad():
        sim_matrix_i2t_fusion = []
        sim_matrix_t2i_fusion = []
        sim_matrix_i2t_pose = []
        sim_matrix_t2i_pose = []
        sim_matrix_i2t_rgb = []
        sim_matrix_t2i_rgb = []
        for idx1, b1 in enumerate(batch_list_v):
            video_mask = b1
            visual_output_pose = batch_visual_output_pose_list[idx1]
            visual_output_rgb = batch_visual_output_rgb_list[idx1]
            # visual_output_cls=visual_cls[idx1]
            each_row_fusion = []
            each_row_fusion_t2i=[]
            each_row_pose = []
            each_row_pose_t2i=[]
            each_row_rgb = []
            each_row_rgb_t2i=[]
            for idx2, b2 in enumerate(batch_list_t):
                text_mask = b2
                sequence_output = batch_sequence_output_list[idx2]
                # sequence_output_cls = sequence_cls[idx2]

                I2T_sim_fusion, T2I_sim_fusion, I2T_sim_pose, T2I_sim_pose, I2T_sim_rgb, T2I_sim_rgb, *tmp = model.get_similarity_logits(sequence_output, visual_output_pose, visual_output_rgb, text_mask, video_mask,
                                                                         loose_type=model.loose_type,is_train=True)
                I2T_sim_fusion = I2T_sim_fusion.cpu().detach().numpy()
                T2I_sim_fusion = T2I_sim_fusion.cpu().detach().numpy()
                I2T_sim_pose = I2T_sim_pose.cpu().detach().numpy()
                T2I_sim_pose = T2I_sim_pose.cpu().detach().numpy()
                I2T_sim_rgb = I2T_sim_rgb.cpu().detach().numpy()
                T2I_sim_rgb = T2I_sim_rgb.cpu().detach().numpy()
                
                each_row_fusion.append(I2T_sim_fusion*dual_mix+T2I_sim_fusion*(1-dual_mix))
                each_row_fusion_t2i.append(I2T_sim_fusion*dual_mix+T2I_sim_fusion*(1-dual_mix))
                each_row_pose.append(I2T_sim_pose*dual_mix+T2I_sim_pose*(1-dual_mix))
                each_row_pose_t2i.append(I2T_sim_pose*dual_mix+T2I_sim_pose*(1-dual_mix))
                each_row_rgb.append(I2T_sim_rgb*dual_mix+T2I_sim_rgb*(1-dual_mix))
                each_row_rgb_t2i.append(I2T_sim_rgb*dual_mix+T2I_sim_rgb*(1-dual_mix))


            each_row_i2t_fusion = np.concatenate(tuple(each_row_fusion), axis=-1)
            each_row_t2i_fusion = np.concatenate(tuple(each_row_fusion_t2i), axis=-1)
            each_row_i2t_pose = np.concatenate(tuple(each_row_pose), axis=-1)
            each_row_t2i_pose = np.concatenate(tuple(each_row_pose_t2i), axis=-1)
            each_row_i2t_rgb = np.concatenate(tuple(each_row_rgb), axis=-1)
            each_row_t2i_rgb = np.concatenate(tuple(each_row_rgb_t2i), axis=-1)

            sim_matrix_i2t_fusion.append(each_row_i2t_fusion)
            sim_matrix_t2i_fusion.append(each_row_t2i_fusion)
            sim_matrix_i2t_pose.append(each_row_i2t_pose)
            sim_matrix_t2i_pose.append(each_row_t2i_pose)
            sim_matrix_i2t_rgb.append(each_row_i2t_rgb)
            sim_matrix_t2i_rgb.append(each_row_t2i_rgb)

        return sim_matrix_i2t_fusion, sim_matrix_t2i_fusion, sim_matrix_i2t_pose, sim_matrix_t2i_pose, sim_matrix_i2t_rgb, sim_matrix_t2i_rgb


def eval_epoch(args, model, test_dataloader, device, n_gpu,istrain):
    torch.cuda.empty_cache()
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()

    start_time = time.time()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_visual_output_pose_list, batch_visual_output_rgb_list = [], []
        batch_sequence_output_list = []

        total_text_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        for bid, batch in enumerate(test_dataloader):

            batch = {k:v.to(device) for k,v in batch.items()}
            sample = batch
            right_batch = {'pose':sample['right_pose']}
            left_batch = {'pose':sample['left_pose']}
            body_batch = {'pose':sample['body_pose'], 'clips_start':sample['body_clips_start'], 'mask':sample['body_mask'], 'rgb':sample['RGB_feature']}
            video_mask = sample['body_mask']
            
            input_ids = sample['pairs_text']
            input_mask = sample['pairs_mask']
            segment_ids = sample['pairs_segment']


            b, *_t = input_ids.shape
            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                mask, visual_hidden_pose, visual_hidden_rgb = model.get_visual_output(right_batch, left_batch, body_batch, shaped=True, get_hidden=True)
                batch_visual_output_pose_list.append(visual_hidden_pose)
                batch_visual_output_rgb_list.append(visual_hidden_rgb)
                batch_list_v.append(mask)

                s_, e_ = total_text_num, total_text_num + b
                #cut_off_points_ len为 sentence num，若video为多个，itm-s_将为不连续的值，也就是仅取一个sentence。
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    input_ids, segment_ids, input_mask = input_ids[filter_inds, ...], segment_ids[filter_inds, ...], segment_ids[input_mask, ...]
                    text_mask,sequence_output= model.get_sequence_output(input_ids, segment_ids, input_mask,get_hidden=True)
                    batch_sequence_output_list.append(sequence_output)
                    batch_list_t.append(text_mask)
                total_text_num += b
            else:
                ValueError("multi_sentence_")

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------


    #########################################


        sim_matrix_i2t_fusion, sim_matrix_t2i_fusion, sim_matrix_i2t_pose, sim_matrix_t2i_pose, sim_matrix_i2t_rgb, sim_matrix_t2i_rgb = _run_on_single_gpu_new_mix(model,  batch_list_v,batch_list_t, batch_sequence_output_list, batch_visual_output_pose_list,\
                                                           batch_visual_output_rgb_list,is_train=False,dual_mix=args.dual_mix)
        
        sim_matrix_i2t_fusion = np.concatenate(tuple(sim_matrix_i2t_fusion), axis=0)
        sim_matrix_t2i_fusion = np.concatenate(tuple(sim_matrix_t2i_fusion), axis=0)
        sim_matrix_i2t_pose = np.concatenate(tuple(sim_matrix_i2t_pose), axis=0)
        sim_matrix_t2i_pose = np.concatenate(tuple(sim_matrix_t2i_pose), axis=0)
        sim_matrix_i2t_rgb = np.concatenate(tuple(sim_matrix_i2t_rgb), axis=0)
        sim_matrix_t2i_rgb = np.concatenate(tuple(sim_matrix_t2i_rgb), axis=0)

        if args.do_eval:
            logger.info(f"before reshape, sim matrix save to {args.output_dir}")
            pkl.dump(sim_matrix_i2t_fusion, open(os.path.join(args.output_dir, "sim_matrix_i2t_before.pkl"), 'wb'))
            pkl.dump(sim_matrix_t2i_fusion, open(os.path.join(args.output_dir, "sim_matrix_t2i_before.pkl"), 'wb'))

        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix_i2t_fusion.shape[0], sim_matrix_i2t_fusion.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])

        sim_matrix_new_i2t_fusion = []
        sim_matrix_new_t2i_fusion = []
        sim_matrix_new_i2t_pose = []
        sim_matrix_new_t2i_pose = []
        sim_matrix_new_i2t_rgb = []
        sim_matrix_new_t2i_rgb = []

        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            # print(e_-s_,max_length-e_+s_)
            # shape: [max_length, sim_matrix_i2t.shape[1]]
            new_matrix_n=np.concatenate((sim_matrix_i2t_fusion[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix_i2t_fusion.shape[1]), -np.inf)), axis=0)
            new_matrix_n_t2i=np.concatenate((sim_matrix_t2i_fusion[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix_t2i_fusion.shape[1]), -np.inf)), axis=0)
            sim_matrix_new_i2t_fusion.append(new_matrix_n)
            sim_matrix_new_t2i_fusion.append(new_matrix_n_t2i)

            new_matrix_n=np.concatenate((sim_matrix_i2t_pose[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix_i2t_pose.shape[1]), -np.inf)), axis=0)
            new_matrix_n_t2i=np.concatenate((sim_matrix_t2i_pose[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix_t2i_pose.shape[1]), -np.inf)), axis=0)
            sim_matrix_new_i2t_pose.append(new_matrix_n)
            sim_matrix_new_t2i_pose.append(new_matrix_n_t2i)

            new_matrix_n=np.concatenate((sim_matrix_i2t_rgb[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix_i2t_rgb.shape[1]), -np.inf)), axis=0)
            new_matrix_n_t2i=np.concatenate((sim_matrix_t2i_rgb[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix_t2i_rgb.shape[1]), -np.inf)), axis=0)
            sim_matrix_new_i2t_rgb.append(new_matrix_n)
            sim_matrix_new_t2i_rgb.append(new_matrix_n_t2i)

        # shape: [sim_matrix_i2t.shape[1], max_length, sim_matrix_i2t.shape[1]]
        sim_matrix_i2t_fusion = np.stack(tuple(sim_matrix_new_i2t_fusion), axis=0)
        sim_matrix_t2i_fusion = np.stack(tuple(sim_matrix_new_t2i_fusion), axis=0)
        sim_matrix_i2t_pose = np.stack(tuple(sim_matrix_new_i2t_pose), axis=0)
        sim_matrix_t2i_pose = np.stack(tuple(sim_matrix_new_t2i_pose), axis=0)
        sim_matrix_i2t_rgb = np.stack(tuple(sim_matrix_new_i2t_rgb), axis=0)
        sim_matrix_t2i_rgb = np.stack(tuple(sim_matrix_new_t2i_rgb), axis=0)

        if args.do_eval:
            logger.info(f"after reshape, sim matrix save to {args.output_dir}")
            pkl.dump(sim_matrix_i2t_fusion, open(os.path.join(args.output_dir, "sim_matrix_i2t_after.pkl"), 'wb'))
            pkl.dump(sim_matrix_t2i_fusion, open(os.path.join(args.output_dir, "sim_matrix_t2i_after.pkl"), 'wb'))

        logger.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix_i2t_fusion.shape[0], sim_matrix_i2t_fusion.shape[1], sim_matrix_i2t_fusion.shape[2]))

        vt_metrics_fusion = tensor_text_to_video_metrics(sim_matrix_i2t_fusion)
        tv_metrics_fusion = compute_metrics(tensor_video_to_text_sim(sim_matrix_t2i_fusion))
        logger.info("Mix_Text-to-Video:")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics_fusion['R1'], tv_metrics_fusion['R5'], tv_metrics_fusion['R10'], tv_metrics_fusion['MR'], tv_metrics_fusion['MeanR']))
        logger.info("Mix_Video-to-Text:")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(vt_metrics_fusion['R1'], vt_metrics_fusion['R5'], vt_metrics_fusion['R10'], vt_metrics_fusion['MR'], vt_metrics_fusion['MeanR']))

        vt_metrics = tensor_text_to_video_metrics(sim_matrix_i2t_pose)
        tv_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix_t2i_pose))
        logger.info("Pos_Text-to-Video:")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        logger.info("Pos_Video-to-Text:")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))
        
        vt_metrics = tensor_text_to_video_metrics(sim_matrix_i2t_rgb)
        tv_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix_t2i_rgb))
        logger.info("RGB_Text-to-Video:")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
        logger.info("RGB_Video-to-Text:")
        logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                    format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))


    R1 = tv_metrics_fusion['R1']

    end_time = time.time() 
    print("操作耗时：", (end_time - start_time)*1000, "毫秒")
    torch.cuda.empty_cache()
    return R1

def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    assert  args.task_type == "retrieval"
    model = init_model(args, device)
            
    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["dev"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None and args.local_rank == 0:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    ## ####################################
    # train and eval
    ## ####################################


    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = -0.00001
        best_epoch=0
        ## ##############################################################
        # resume optimizer state besides loss to continue trainxz
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
        
        global_step = 0
        loss_record=[]
        acc_record=[]
        for epoch in range(resumed_epoch, args.epochs):
            if args.distributed==True:
                train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0 and (epoch+1)%10==0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(epoch, args, model, optimizer, tr_loss, type_name="")

            if args.local_rank == 0:
                R1 = eval_epoch(args, model, test_dataloader, device, n_gpu,False)
                if best_score <= R1:
                    best_score = R1
                    best_epoch=epoch
                    best_output_model_file = save_best_model(epoch, args, model, optimizer, tr_loss, type_name="")
                logger.info("The best model is: {}{}, the R1 is: {:.4f}, the model file is {}".format(args.output_dir,best_epoch, best_score, best_output_model_file))
                loss_record.append(tr_loss)
                acc_record.append(R1)
                logger.info(loss_record)
                logger.info(acc_record)

                if epoch == args.stop_epochs:
                    break

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu,False)

if __name__ == "__main__":
    main()
