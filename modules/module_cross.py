from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from .file_utils import cached_path
from .until_config import PretrainedConfig
from .until_module import PreTrainedModel, LayerNorm, ACT2FN
from collections import OrderedDict
from torch import Tensor

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {}
CONFIG_NAME = 'cross_config.json'
WEIGHTS_NAME = 'cross_pytorch_model.bin'


class CrossConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `CrossModel`.
    """
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    config_name = CONFIG_NAME
    weights_name = WEIGHTS_NAME
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs CrossConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `CrossModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `CrossModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.n_head = n_head

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor):
        attn_mask_ = attn_mask.repeat_interleave(self.n_head, dim=0)
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, para_tuple: tuple):
        # x: torch.Tensor, attn_mask: torch.Tensor
        # print(para_tuple)
        x, attn_mask = para_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, attn_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):

        x = x.reshape(x.shape[0], x.shape[1], -1)
        return self.resblocks((x, attn_mask))[0]

class CrossEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(CrossEmbeddings, self).__init__()

        self.position_embeddings = nn.Embedding(66, config.hidden_size)
        # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, concat_embeddings, concat_type=None):

        batch_size, seq_length = concat_embeddings.size(0), concat_embeddings.size(1)
        # if concat_type is None:
        #     concat_type = torch.zeros(batch_size, concat_type).to(concat_embeddings.device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=concat_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(concat_embeddings.size(0), -1)

        # token_type_embeddings = self.token_type_embeddings(concat_type)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = concat_embeddings + position_embeddings # + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CrossPooler(nn.Module):
    def __init__(self, config):
        super(CrossPooler, self).__init__()
        self.ln_pool = LayerNorm(config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = QuickGELU()

    def forward(self, hidden_states, hidden_mask):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hidden_states = self.ln_pool(hidden_states)
        pooled_output = hidden_states[:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CrossModel(PreTrainedModel):

    def initialize_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def __init__(self, config):
        super(CrossModel, self).__init__(config)

        self.embeddings = CrossEmbeddings(config)

        transformer_width = config.hidden_size
        transformer_layers = config.num_hidden_layers
        transformer_heads = config.num_attention_heads
        self.transformer = Transformer(width=transformer_width, layers=transformer_layers, heads=transformer_heads,)
        self.pooler = CrossPooler(config)
        self.apply(self.init_weights)

    def build_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0
        extended_attention_mask = extended_attention_mask.expand(-1, attention_mask.size(1), -1)
        return extended_attention_mask

    def forward(self, concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True):

        if attention_mask is None:
            attention_mask = torch.ones(concat_input.size(0), concat_input.size(1))
        if concat_type is None:
            concat_type = torch.zeros_like(attention_mask)

        extended_attention_mask = self.build_attention_mask(attention_mask)

        embedding_output = self.embeddings(concat_input, concat_type)
        embedding_output = embedding_output.permute(1, 0, 2)  # NLD -> LND
        embedding_output = self.transformer(embedding_output, extended_attention_mask)
        embedding_output = embedding_output.permute(1, 0, 2)  # LND -> NLD

        pooled_output = self.pooler(embedding_output, hidden_mask=attention_mask)

        return embedding_output, pooled_output
    
class Conv_feature_fusion(nn.Module):
    def __init__(self, input_channel, output_channel, dropout=0.0):
        super(Conv_feature_fusion, self).__init__()
        self.conv1d_5 = nn.Sequential(nn.Conv1d(input_channel, output_channel, 5, stride=1, padding=2), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.conv1d_3_1 = nn.Sequential(nn.Conv1d(input_channel, input_channel, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.conv1d_3_2 = nn.Sequential(nn.Conv1d(input_channel, output_channel, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.conv1x1 = nn.Conv1d(output_channel*2, output_channel, 1, stride=1, padding=0)

        self.bn = nn.BatchNorm1d(output_channel)

    def forward(self, pose, rgb):
        
        x = torch.concat((pose, rgb), dim=-1).permute(0,2,1)
        x = torch.cat([self.conv1d_5(x), self.conv1d_3_2(self.conv1d_3_1(x))], dim=1)
        x = self.bn(self.conv1x1(x)).permute(0,2,1)

        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)

    def attention(self, q, kv, attn_mask):

        return self.attn(q, kv, kv, key_padding_mask=attn_mask, need_weights=False)[0]

    def forward(self, x_tuple:tuple):

        q, kv, attn_mask = x_tuple
        q = q + self.attention(self.ln_1(q), self.ln_2(kv), attn_mask)
        q = q + self.mlp(self.ln_3(q))

        return (q, kv, attn_mask)
    
class Cross_Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[CrossAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_mask: torch.Tensor):
        return self.resblocks((q, kv, attn_mask))[0]
    
class RGB_Encoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, input_channel: int):
        super().__init__()
        
        self.hand_l = nn.Linear(input_channel*2, input_channel*2)
        self.body_l = nn.Linear(input_channel, input_channel*2)
        self.pose_l = nn.Linear(input_channel*3, input_channel*2)
        self.crossformer = Cross_Transformer(width, layers, heads, None)

    def forward(self, pose: torch.Tensor, rgb: torch.Tensor):
        
        hand, body = torch.split(pose, [1024, 512], dim=-1)
        rgb = rgb.squeeze().permute(0, 2, 1)
        B, L, C = rgb.shape
        rgb = rgb.reshape(B*L, 1, C).permute(1, 0, 2)

        hand = self.hand_l(hand).reshape(B*L, 1, C)
        pose = self.pose_l(pose).reshape(B*L, 1, C)
        body = self.body_l(body).reshape(B*L, 1, C)
        pose = torch.concat((body, hand, pose), dim=-2).permute(1, 0, 2)

        attn_mask = torch.zeros((B*L, pose.shape[0])).to(device=pose.device).bool()

        rgb = self.crossformer(rgb, pose, attn_mask) + rgb
        
        return rgb.permute(1, 0, 2).reshape(B, L, C).permute(0, 2, 1).unsqueeze(-1)
    



