import torch
import torch.nn as nn
from modules.modeling_gcn import GCN_Embed
from modules.modeling_gcn import TemporalConvNetBlock, TemporalConvNetBlock_fusion
import numpy as np
import math
import torch.nn.functional as F
import logging
logger = logging.getLogger(__name__)


def init_sign_model(args):

    if args.init_sign_model:
        logger.info('loading signbert checkpoint {}'.format(args.init_sign_model))
        checkpoint = torch.load(args.init_sign_model, map_location='cpu')
        model_state_dict = checkpoint['state_dict']['GCN_Transform'] if 'GCN_Transform' in checkpoint['state_dict'] else checkpoint['state_dict']
    else:
        model_state_dict = None

    model = Sign_Bert.from_pretrained(state_dict=model_state_dict, opt=args)

    return model

class Sign_Bert(torch.nn.Module):
    def __init__(self, opt, stgcn=False):
        super(Sign_Bert, self).__init__()
        self.stgcn_only = stgcn
        self.embed = GCN_Embed(gcn_args=opt, d_model=opt.hidden_dim, dropout=opt.dropout)

        self.GCN_Conv = TemporalConvNetBlock_fusion(num_inputs=opt.hidden_dim*3, num_outputs=opt.hidden_dim*3)

    @classmethod
    def from_pretrained(cls, state_dict=None, opt=None):
        model = cls(opt, stgcn=False)

        if state_dict is not None:
                           
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(
                    state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + '.')

            load(model, prefix='')

            if opt.local_rank == 0:
                logger.info("-" * 20)
                if len(missing_keys) > 0:
                    logger.info("Weights of {} not initialized from pretrained model: {}"
                                .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
                if len(unexpected_keys) > 0:
                    logger.info("Weights from pretrained model not used in {}: {}"
                                .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
                if len(error_msgs) > 0:
                    logger.error("Weights from pretrained model cause errors in {}: {}"
                                .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    def forward(self, pose_data):
        pose_data = self.embed(pose_data)
        if self.stgcn_only is True:
            return pose_data['feat']

        pose_feat = self.GCN_Conv(pose_data['feat'])
        
        return pose_feat
    
    def gcn_emb(self, pose_data):
        pose_data = self.embed.embedding(pose_data)

        return pose_data
    
    def sign_conv(self, pose_feat):

        pose_feat = pose_feat.permute(0, 2, 1)
        pose_feat = self.GCN_Conv(pose_feat)
        pose_feat = pose_feat.permute(0, 2, 1)
        return pose_feat
    
class Sign_Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, dim, n_heads, dim_ff, blocks, dropout=0.1):
        super(Sign_Transformer, self).__init__()
        self.blocks = nn.ModuleList([Block(dim, n_heads, dim_ff, dropout) for _ in range(blocks)])
        self.norm = LayerNorm(dim)

    def forward(self, h, mask):
        h = self.norm(h)
        for block in self.blocks:
            h = block(h, mask)

        return h
    
class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
    
class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, dim, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(dim, n_heads, dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = LayerNorm(dim)
        self.pwff = PositionWiseFeedForward(dim, dim_ff)
        self.norm2 = LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.scores = None  # for visualization
        self.n_heads = n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            if len(mask.shape) == 2:
                mask = mask[:, None, None, :].float()
            else:
                mask = mask[:, None, :, :].float()
            # scores = scores.masked_fill(mask==0, -1e9)
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, dim, dim_ff):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))