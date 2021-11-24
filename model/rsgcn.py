import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels,branch=4,stride=1):
        super(unit_tcn, self).__init__()
        self.branch = branch
        b_in = in_channels//branch
        b_out = out_channels//branch
        branch_3 = nn.Sequential(nn.Conv2d(b_in, b_out, kernel_size=(3, 1), padding=(1, 0), stride=(stride, 1),bias=False),
                                      nn.BatchNorm2d(b_out))
        branch_5 = nn.Sequential(nn.Conv2d(b_in, b_out, kernel_size=(5, 1), padding=(2, 0), stride=(stride, 1),bias=False),
                                      nn.BatchNorm2d(b_out))
        branch_7 = nn.Sequential(nn.Conv2d(b_in, b_out, kernel_size=(7, 1), padding=(3, 0), stride=(stride, 1),bias=False),
                                      nn.BatchNorm2d(b_out))
        branch_9 = nn.Sequential(nn.Conv2d(b_in, b_out, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1),bias=False),
                                      nn.BatchNorm2d(b_out))
        self.branches = nn.ModuleList([branch_3,branch_5,branch_7,branch_9])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x):
        x = list(x.chunk(dim=1,chunks=4))
        for i in range(len(x)):
            x[i] = self.branches[i](x[i])
        x = torch.cat(x,dim=1)
        return x







class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3,gate=0.1,start=0.9):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.num_subset = num_subset
        self.conv_c = nn.Conv2d(in_channels, inter_channels, 1)
        self.conv_d = nn.Conv2d(inter_channels*num_subset, out_channels, 1,bias=False)
        self.gate = nn.Parameter((gate*torch.ones(1)),requires_grad=False)
        self.prepare_A(inter_channels,A,gate,start)
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        # bn_init(self.bn,1e-6)

    def prepare_A(self,inter_channels,A,gate,start,gap=0.05):
        A = np.repeat(np.expand_dims(A, 1), inter_channels, axis=1)
        rand_start = start*gate+(1-start-gap)*gate * np.random.random(size=A.shape)
        A = np.where(A > 0, A, rand_start).astype(np.float32)
        self.PA = nn.Parameter(torch.from_numpy(A))


    def forward(self, x,s):
        N, C, T, V = x.size()
        # A = self.A.cuda(x.get_device())
        A = torch.where(self.PA>self.gate,self.PA,torch.zeros_like(self.PA))
        s = s+(torch.where(self.PA<=self.gate,self.gate-self.PA,torch.zeros_like(self.PA))).sum()
        y = self.conv_c(x).view(N,1,self.inter_c,T,V)
        y = torch.matmul(y, A).view(N, -1, T, V)
        y = self.conv_d(y)
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y),s






class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x,s):
        res = self.residual(x)
        x, s = self.gcn1(x,s)
        x = self.tcn1(x)+res
        return self.relu(x),s






class spatial_temporal_transformer(nn.Module):
    def __init__(self,in_channels,coff=2,heads=4):
        super(spatial_temporal_transformer,self).__init__()
        inter_channels = in_channels//coff
        head_c = inter_channels//heads
        self.inter_c = inter_channels
        self.heads = heads
        self.head_c =head_c
        self.spatial_qconv = nn.Conv2d(in_channels,inter_channels,1)
        self.spatial_kconv = nn.Conv2d(in_channels, inter_channels, 1)
        self.spatial_vconv = nn.Conv2d(in_channels, inter_channels, 1)
        self.spatial_upconv = nn.Conv2d(inter_channels, in_channels, 1,bias=False)

        self.temporal_qconv = nn.Conv2d(in_channels, inter_channels, 1)
        self.temporal_kconv = nn.Conv2d(in_channels, inter_channels, 1)
        self.temporal_vconv = nn.Conv2d(in_channels, inter_channels, 1)
        self.temporal_upconv = nn.Conv2d(inter_channels, in_channels, 1,bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn1,1e-6)
        bn_init(self.bn2,1e-6)
        self.soft_max = nn.Softmax(-2)
        self.relu = nn.ReLU()


    def transfer(self, x, dim):
        N, C, T, V = x.size()
        head_c = self.head_c
        heads = self.heads
        if dim == 'spatial':
            k = self.spatial_kconv(x).view(N, heads, head_c*T, V).permute(0,1,3,2).contiguous()
            q = self.spatial_qconv(x).view(N, heads, head_c*T, V)
            v = self.spatial_vconv(x).view(N, heads, head_c*T, V)
        if dim == 'temporal':
            k = self.temporal_kconv(x).view(N, heads, head_c, T, V).permute(0,1,3,2,4).contiguous()
            k = k.view(N, heads, T, head_c*V)
            q = self.temporal_qconv(x).permute(0, 1, 3, 2).contiguous().view(N, heads, head_c*V, T)
            v = self.temporal_vconv(x).permute(0, 1, 3, 2).contiguous().view(N, heads, head_c*V, T)
        norm_k = torch.sqrt(torch.sum(k ** 2, -1, keepdim=True))
        norm_q = torch.sqrt(torch.sum(q ** 2, -2, keepdim=True))
        q = torch.matmul(k, q) / (norm_k * norm_q)
        q = 1 / torch.acos(q)
        q = q - q * torch.eye(q.size(-1)).unsqueeze(0).cuda(k.get_device())
        q = self.soft_max(q)
        if dim == 'spatial':
            v = torch.matmul(v, q).view(N, self.inter_c, T, V)
            v = self.bn1(self.spatial_upconv(v))
        if dim == 'temporal':
            v = torch.matmul(v, q).view(N, self.inter_c, V, T).permute(0, 1, 3, 2)
            v = self.bn2(self.temporal_upconv(v))
        return  v
    def forward(self, x):
        # y = self.transfer(x,dim='temporal')
        # x = y+x
        # y = self.transfer(x,dim='spatial')
        # x = y+x
        # x = self.relu(x)
        y1 = self.transfer(x,dim='temporal')
        y2 = self.transfer(x,dim='spatial')
        x = y1+y2+x
        x = self.relu(x)
        return  x






class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(),
                 in_channels=3,drop_out_rate=0,alpha=2.2e-5):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        if num_point == 20:#,nwucla
            trans_coff = 2
            trans_heads= 2
        elif num_point == 25:  #ntu 
            trans_coff = 4
            trans_heads=1
        elif num_point == 18:
            trans_coff = 2
            trans_heads = 4
        else:
            raise  AttributeError
        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = spatial_temporal_transformer(128,trans_coff,trans_heads)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = spatial_temporal_transformer(256,trans_coff,trans_heads)
        self.l10 = TCN_GCN_unit(256, 256, A)
        self.drop_out = nn.Dropout(drop_out_rate)
        self.fc = nn.Linear(256, num_class)
        self.alpha = nn.Parameter((alpha*torch.ones(1)),requires_grad=False)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        s = 0
        x,s = self.l1(x,s)
        x,s = self.l2(x,s)
        x,s = self.l3(x,s)
        x,s = self.l4(x,s)
        x,s = self.l5(x,s)
        x = self.l6(x)
        x,s = self.l7(x,s)
        x,s = self.l8(x,s)
        x = self.l9(x)
        x,s = self.l10(x,s)
        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        return self.fc(x),self.alpha*s
