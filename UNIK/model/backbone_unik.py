import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

# Temporal unit
class T_LSU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, autopad=True):
        super(T_LSU, self).__init__()

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        bn_init(self.bn, 1)
        if autopad:
            pad = int(( kernel_size - 1) * dilation // 2)
        else:
            pad = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))
        conv_init(self.conv)

    def forward(self, x):
        conv_part = self.bn(self.conv(x))
        return conv_part

# Spatial unit
class S_LSU(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints, num_heads=8, coff_embedding=4, bias=True):
        super(S_LSU, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.dropout = nn.Dropout(0.3)
        self.inter_c = inter_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.DepM = nn.Parameter(torch.Tensor(num_heads, num_joints, num_joints))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_joints))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # Attention
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_heads):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

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
        bn_init(self.bn, 1e-6)
        for i in range(self.num_heads):
            conv_branch_init(self.conv_d[i], self.num_heads)


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.DepM, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.DepM)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, x):
        N, C, T, V = x.size()

        W = self.DepM
        B = self.bias
        y = None
        for i in range(self.num_heads):

            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N tV tV

            A1 = W[i] + A1
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i]((torch.matmul(A2, A1)).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y).view(N, -1, T, V)



class ST_block(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=25, num_heads=3, stride=1, dilation=1, autopad=True, residual=True):
        super(ST_block, self).__init__()
        self.s_unit = S_LSU(in_channels, out_channels, num_joints, num_heads)
        self.t_unit = T_LSU(out_channels, out_channels, stride=stride, dilation=dilation, autopad=autopad)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pad = 0
        if not autopad:
            self.pad = (9 - 1) * dilation // 2

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = T_LSU(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        s_out = self.dropout(self.s_unit(x))
        x = self.dropout(self.t_unit(s_out) + self.residual(x[:, :, self.pad : x.shape[2] - self.pad, :]))
        return self.relu(x)


class UNIK(nn.Module):
    def __init__(self, num_class=60, num_joints=20, num_person=1, num_heads=3, in_channels=3):
        super(UNIK, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.l1 = ST_block(in_channels, 64, num_joints, residual=False)

        self.l2 = ST_block(64, 64, num_joints, num_heads, dilation=1) #3
        self.l3 = ST_block(64, 64, num_joints, num_heads, dilation=1)  #3
        self.l4 = ST_block(64, 64, num_joints, num_heads, dilation=1)   #3

        self.l5 = ST_block(64, 128, num_joints, num_heads, stride=2)
        self.l6 = ST_block(128, 128, num_joints, num_heads)
        self.l7 = ST_block(128, 128, num_joints, num_heads)

        self.l8 = ST_block(128, 256, num_joints, num_heads, stride=2)
        self.l9 = ST_block(256, 256, num_joints, num_heads)
        self.l10 = ST_block(256, 256, num_joints, num_heads)
        self.l11 = ST_block(256, 128, num_joints, num_heads)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        N, C, T, V, M = x.size() #64 3 16 20 1
        x = x.squeeze(-1)

        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        x = self.l1(x)
        #x = self.l2(x)
        #x = self.l3(x)
        #x = self.l4(x)
        x = self.l5(x)
        #x = self.l6(x)
        #x = self.l7(x)
        x = self.l8(x)
        #x = self.l9(x)
        #x = self.l10(x)
        x = self.l11(x)

        x = x.view(N, x.size(1), -1)
        x = x.mean(-1)

        x = self.fc(x)

        return x
