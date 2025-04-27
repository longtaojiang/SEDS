import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules.modeling_graph import Graph, Graph_pool

class TemporalConvNetBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs, dropout=0.0):
        super(TemporalConvNetBlock, self).__init__()
        self.conv1d_5 = nn.Sequential(nn.Conv1d(num_inputs, num_outputs, 5, stride=1, padding=2), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.conv1d_3_1 = nn.Sequential(nn.Conv1d(num_inputs, num_outputs, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.conv1d_3_2 = nn.Sequential(nn.Conv1d(num_inputs, num_outputs, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.conv1x1 = nn.Conv1d(num_outputs*2, num_outputs, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm1d(num_outputs)

    def forward(self, x):
        x = torch.cat([self.conv1d_5(x), self.conv1d_3_2(self.conv1d_3_1(x))], dim=1)

        x = self.bn(self.conv1x1(x))
        x = self.relu(x)
        return x

class TemporalConvNetBlock_fusion(nn.Module):
    def __init__(self, num_inputs, num_outputs, dropout=0.0):
        super(TemporalConvNetBlock_fusion, self).__init__()
        self.conv1d_5 = nn.Sequential(nn.Conv1d(num_inputs, num_outputs, 5, stride=1, padding=2), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.conv1d_3_1 = nn.Sequential(nn.Conv1d(num_inputs, num_inputs, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.conv1d_3_2 = nn.Sequential(nn.Conv1d(num_inputs, num_outputs, 3, stride=1, padding=1), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.conv1x1 = nn.Conv1d(num_outputs*2, num_outputs, 1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        self.bn = nn.BatchNorm1d(num_outputs)

    def forward(self, x):
        x = torch.cat([self.conv1d_5(x), self.conv1d_3_2(self.conv1d_3_1(x))], dim=1)

        x = self.bn(self.conv1x1(x))
        x = self.relu(x)
        return x
    
class ConvTemporalGraphical(nn.Module):

    """The basic module for applying a graph convolution.
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels,1,  T*V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, T*V, T*V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, 1, T*V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, T*V, T*V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the kernel size`,
            :math:`T` is a length of sequence,
            :math:`V` is the number of graph nodes per frame.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1), 
            dilation=(t_dilation, 1), #Spacing between kernel elements, dilated Convolution
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A
    
class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size :number of the node clusters
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, 1, T*V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, T*V, T*V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, 1, T*V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, T*V, T*V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the kernel size
            :math:`T` is a length of sequence,
            :math:`V` is the number of graph nodes of each frame.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0.05,
                 residual=True):

        super(st_gcn,self).__init__()
        self.inplace = True

        self.momentum = 0.1
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)

        self.tcn = nn.Sequential(

            nn.BatchNorm2d(out_channels, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.05),
            nn.Conv2d(
                out_channels,
                out_channels,
                (1, 1),
                (stride, 1),
                padding = 0,
            ),
            nn.BatchNorm2d(out_channels, momentum=self.momentum),
            nn.Dropout(dropout, inplace=self.inplace),


        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels, momentum=self.momentum),
            )

        self.relu = nn.ReLU(inplace=self.inplace)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A.to(x.device))

        x = self.tcn(x) + res

        return self.relu(x), A

class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=600):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self,x):
        '''
        :param x: size (B, T, C)
        :return: x+position_encoding(x)
        '''
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class GCN_Embed(nn.Module):
    def __init__(self, gcn_args, d_model=512, dropout=0.1):
        super(GCN_Embed, self).__init__()
        gcn_args.layout_encoder = 'stb'
        self.st_gcn_hand = ST_GCN_Model(opt=gcn_args)
        gcn_args.layout_encoder = 'body'
        self.st_gcn_body = ST_GCN_Model(opt=gcn_args)
        self.d_model = d_model

    def forward(self, pose):
        right_pose = pose['right']
        left_pose = pose['left']
        body_pose = pose['body']
        right_feat = self.st_gcn_hand(right_pose)
        left_feat = self.st_gcn_hand(left_pose)
        body_feat = self.st_gcn_body(body_pose)
        pose_feat = torch.cat([left_feat, right_feat, body_feat], dim=-1)
        pose['feat'] = pose_feat
        
        return pose
    
    def embedding(self, pose):
        right_pose = pose['right']
        left_pose = pose['left']
        body_pose = pose['body']
        right_feat = self.st_gcn_hand(right_pose)
        left_feat = self.st_gcn_hand(left_pose)
        body_feat = self.st_gcn_body(body_pose)
        pose_feat = torch.cat([left_feat, right_feat, body_feat], dim=-1)
        pose['feat'] = pose_feat
        return pose
    
class ST_GCN_Model(nn.Module):
    """
    Args:
        in_channels (int): Number of channels in the input data
        cat: True: concatinate coarse and fine features
            False: add coarse and fine features
        pad:
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes for each frame,
            :math:`M_{in}` is the number of instance in a frame. (In this task always equals to 1)
    Return:
        out_all_frame: True: return all frames 3D results
                        False: return target frame result
        x_out: final output.
    """

    def __init__(self, opt):
        super(ST_GCN_Model,self).__init__()

        # load graph
        self.momentum = 0.1
        self.in_channels = opt.in_channels
        self.layout = opt.layout_encoder
        self.strategy = opt.strategy
        self.cat = True
        self.inplace = True
        self.pad = opt.temporal_pad

        # original graph
        self.graph = Graph(self.layout, self.strategy, pad=opt.temporal_pad)
        # get adjacency matrix of K clusters
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False).cuda()  # K
        self.register_buffer('A', A)
        # pooled graph
        self.graph_pool = Graph_pool(self.layout, self.strategy, pad=opt.temporal_pad)
        A_pool = torch.tensor(self.graph_pool.A, dtype=torch.float32, requires_grad=False).cuda()
        self.register_buffer('A_pool', A_pool)

        # build networks, K=4 subset: self, to close, to far, sym same joints
        kernel_size = self.A.size(0)
        kernel_size_pool = self.A_pool.size(0)

        self.data_bn = nn.BatchNorm1d(self.in_channels * self.graph.num_node_each, self.momentum)

        self.inter_channels = [128, 128, 256]

        self.fc_out = self.inter_channels[-1]
        self.fc_unit = 512

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(self.in_channels, self.inter_channels[0], kernel_size, residual=False),
            st_gcn(self.inter_channels[0], self.inter_channels[1], kernel_size),
            st_gcn(self.inter_channels[1], self.inter_channels[2], kernel_size),
        ))


        self.st_gcn_pool = nn.ModuleList((
            st_gcn(self.inter_channels[-1], self.fc_unit, kernel_size_pool),
            st_gcn(self.fc_unit, self.fc_unit, kernel_size_pool),
        ))


        self.conv4 = nn.Sequential(
            nn.Conv2d(self.fc_unit, self.fc_unit, kernel_size=(1, 1), padding = (0, 0)),
            nn.BatchNorm2d(self.fc_unit, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.25)
        )

        # tcn block
        self.tcn_full_b1 = TemporalConvNetBlock(self.fc_unit, self.fc_unit)
        self.tcn_full_b2 = TemporalConvNetBlock(self.fc_unit, self.fc_unit)

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p,stride=None):
        if max(p) > 1:
            if stride is None:
                x = nn.MaxPool2d(p)(x)  # B x F x V/p
            else:
                x = nn.MaxPool2d(kernel_size=p,stride=stride)(x)  # B x F x V/p
            return x
        else:
            return x


    def forward(self, x):

        batch, sequence, num_joint, coordination = x.size()
        x = x.contiguous().view(batch * sequence, 1, num_joint, coordination)
        x = x.permute(0, 3, 1, 2) 
        x = x.unsqueeze(-1)

        # data normalization, T there is time connection, but the time_pad = 0 
        N, C, T, V, M= x.size()  #(batch * sequence, coordination, T=1, num_joint, 1)

        x = x.permute(0, 4, 3, 1, 2).contiguous()  #(batch * sequence, 1, num_joint, coordination, 1)
        x = x.view(N * M, V * C, T) #(batch * sequence* 1, num_joint * coordination, 1)

        x = self.data_bn(x) #BN channel = num_joint * coordination
        x = x.view(N, M, V, C, T) #(batch * sequence, 1, num_joint, coordination, 1)
        x = x.permute(0, 1, 3, 4, 2).contiguous() #(batch * sequence, 1, coordination, 1, num_joint)
        x = x.view(N * M, C, 1, -1)  # (batch * sequence * 1, coordination, 1, num_joint)

        # forwad GCN
        gcn_list = list(self.st_gcn_networks)
        for i_gcn, gcn in enumerate(gcn_list):
            x, _ = gcn(x, self.A) # (N * M), C, 1, (T*V) 

        x = x.view(N, -1, T, V)  # N, C, T ,V (batch * sequence, channel, 1, num_joint)

        # Pooling
        for i in range(len(self.graph.part)):
            num_node= len(self.graph.part[i]) #corresponding finger or arm
            x_i = x[:, :, :, self.graph.part[i]] #get the specific part
            x_i = self.graph_max_pool(x_i, (1, num_node))
            x_sub1 = torch.cat((x_sub1, x_i), -1) if i > 0 else x_i # Final to N, C, T, (NUM_SUB_PARTS) (batch * sequence, channel, NUM_SUB_PARTS)

        x_sub1, _ = self.st_gcn_pool[0](x_sub1.view(N, -1, 1, T*len(self.graph.part)), self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
        x_sub1, _ = self.st_gcn_pool[1](x_sub1, self.A_pool.clone())  # N, 512, 1, (T*NUM_SUB_PARTS)
        x_sub1 = x_sub1.view(N, -1, T, len(self.graph.part)) # N, 512, T=1, (NUM_SUB_PARTS)

        x_pool_1 = self.graph_max_pool(x_sub1, (1, len(self.graph.part)))  # N, 512, T, 1
        x_pool_1 = self.conv4(x_pool_1)     # N*T, C, 1, 1
        x_pool_1 = x_pool_1.view(batch, sequence, -1) # N, T, C
        tcn_1_in = x_pool_1.contiguous().permute(0, 2, 1) # N, C, T
        tcn_1_out = self.tcn_full_b1(tcn_1_in) #no need time confusion
        tcn_2_out = self.tcn_full_b2(tcn_1_out) #no need time confusion
        tcn_2_out = tcn_2_out.contiguous().permute(0, 2, 1) # N, T, C

        return tcn_2_out