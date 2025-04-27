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

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MLP_feature_fusion(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(input_channel, input_channel*2)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(input_channel*2, input_channel)),
            ("gelu1", QuickGELU()),
            ("c_proj1", nn.Linear(input_channel, output_channel)),
        ]))
        self.ln = LayerNorm(output_channel)

    def forward(self, pose, rgb, mask):
        
        x = torch.concat((pose, rgb), dim=-1)
        x = self.mlp(x) + pose + rgb
        x = self.ln(x)
        
        return x

class Encoder(nn.Module):
    """
    Base encoder class
    """

    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size

class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)]

class DeformableMultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(
        self,
        num_heads: int,
        size: int,
        dropout: float = 0.1,
        num_keys: int = 5,
    ):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        :param num_keys: The number of observed keys
        """
        super(DeformableMultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)
        self.num_keys = num_keys
        self.sample_offsets = nn.Linear(size, num_heads * self.num_keys)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads
        key_len = k.size(1)
        query_len = q.size(1)

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # batch x num_heads x query_len x num_keys
        offsets = (
            self.sample_offsets(q)
            .reshape(batch_size, -1, num_heads, self.num_keys)
            .transpose(1, 2)
        )

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        location_point = torch.arange(
            0, query_len, device=offsets.device, dtype=torch.float32
        ).unsqueeze(-1)

        left = -self.num_keys // 2
        right = self.num_keys + left
        reference_point = torch.arange(
            left, right, device=offsets.device, dtype=torch.float32
        )
        mask = 1 - mask
        sign_lens = mask.sum(-1).squeeze().float() - 1
        sampling_locations = reference_point + offsets + location_point
        sampling_locations = sampling_locations % sign_lens[:, None, None, None]
        # batch x num_heads x query_len x num_keys
        sampling_locations = sampling_locations / (key_len - 1) * 2 - 1
        y_location = sampling_locations.new_ones(sampling_locations.shape)
        # batch*num_heads x query_len x num_keys x 2
        sampling_locations = torch.stack([sampling_locations, y_location], -1).reshape(
            batch_size * num_heads, query_len, self.num_keys, 2
        )

        # batch*num_heads x head_size x 1 x key_len
        reshaped_k = k.reshape(batch_size * num_heads, 1, -1, self.head_size).permute(
            0, 3, 1, 2
        )
        reshaped_v = v.reshape(batch_size * num_heads, 1, -1, self.head_size).permute(
            0, 3, 1, 2
        )
        
        # batch x num_heads x query_len x num_keys x head_size
        smaple_k = (
            nn.functional.grid_sample(
                input=reshaped_k,
                grid=sampling_locations,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .reshape(batch_size, num_heads, self.head_size, query_len, -1)
            .permute(0, 1, 3, 4, 2)
        )
        smaple_v = (
            nn.functional.grid_sample(
                input=reshaped_v,
                grid=sampling_locations,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .reshape(batch_size, num_heads, self.head_size, query_len, -1)
            .permute(0, 1, 3, 4, 2)
        )

        # compute scores
        q = q / math.sqrt(self.head_size)
        q = q.unsqueeze(3).expand(
            batch_size, num_heads, query_len, self.num_keys, self.head_size
        )
        # batch x num_heads x query_len x num_keys
        scores = (q * smaple_k).sum(-1)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        # if mask is not None:
        #     scores = scores.masked_fill(~mask, float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]

        # batch x num_heads x query_len x head_size
        context = (attention.unsqueeze(-1) * smaple_v).sum(-2)
        context = context.transpose(1, 2).reshape(
            batch_size, -1, num_heads * self.head_size
        )
        output = self.output_layer(context)

        return output

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x

class DeformableTransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self,
        attentions_type: str = "local",
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        num_keys: int = 5,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(DeformableTransformerEncoderLayer, self).__init__()
        self.attention_type = attentions_type
        self.layer_norm = nn.LayerNorm(size, eps=1e-6)

        if attentions_type == "local":
            self.src_src_att = DeformableMultiHeadedAttention(
                num_heads,
                size,
                dropout=dropout,
                num_keys=num_keys,
            )
        
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, q: Tensor, vk: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        q_norm = self.layer_norm(q)
        if self.attention_type == "local":
            h = self.src_src_att(q_norm, vk, vk, mask)
            h = self.dropout(h) + q
        o = self.feed_forward(h)
        return o

    
class Gloss_Fusion_Transformer(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        num_keys: list = [5, 3],
        attentions_type: str = "local",
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param kwargs:
        """
        super(Gloss_Fusion_Transformer, self).__init__()
        assert len(num_keys) == num_layers, "num kyes must equal num layers"
        # build all (num_layers) layers
        self.P2R_layers = nn.ModuleList(
            [
                DeformableTransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    num_keys=num,
                    attentions_type=attentions_type,
                )
                for num in num_keys
            ]
        )
        self.P2R_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.R2P_layers = nn.ModuleList(
            [
                DeformableTransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    num_keys=num,
                    attentions_type=attentions_type,
                )
                for num in num_keys
            ]
        )
        self.R2P_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.pe = PositionalEncoding(hidden_size)

        self.mlp_fusion = MLP_feature_fusion(input_channel=hidden_size*2, output_channel=hidden_size)

        self._output_size = hidden_size

    def forward(
        self, pose_mbed_src: Tensor, RGB_mbed_src: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """

        pose_ori = pose_mbed_src.clone()
        RGB_ori = RGB_mbed_src.clone()

        pose_mbed_src = self.pe(pose_mbed_src)
        RGB_mbed_src = self.pe(RGB_mbed_src)

        pose_mbed_src_ori = pose_mbed_src.clone()
        RGB_mbed_src_ori = RGB_mbed_src.clone()
        pose_mbed_src = self.P2R_layers[0](pose_mbed_src, RGB_mbed_src_ori, mask)
        RGB_mbed_src = self.R2P_layers[0](RGB_mbed_src, pose_mbed_src_ori, mask)

        pose_mbed_src_ori = pose_mbed_src.clone()
        RGB_mbed_src_ori = RGB_mbed_src.clone()
        pose_mbed_src = self.P2R_layers[1](pose_mbed_src, RGB_mbed_src_ori, mask)
        RGB_mbed_src = self.R2P_layers[1](RGB_mbed_src, pose_mbed_src_ori, mask)

        pose_mbed_src = self.P2R_layer_norm(pose_mbed_src)
        RGB_mbed_src = self.R2P_layer_norm(RGB_mbed_src)

        return self.mlp_fusion(pose_mbed_src, RGB_mbed_src, mask) + pose_ori + RGB_ori


class DeformableTransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        num_keys: list,
        attentions_type: str = "local",
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(DeformableTransformerEncoder, self).__init__()
        assert len(num_keys) == num_layers, "num kyes must equal num layers"
        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                DeformableTransformerEncoderLayer(
                    size=hidden_size,
                    ff_size=ff_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    num_keys=num,
                    attentions_type=attentions_type,
                )
                for num in num_keys
            ]
        )

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)

        self._output_size = hidden_size

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        x = self.pe(embed_src)

        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x), None