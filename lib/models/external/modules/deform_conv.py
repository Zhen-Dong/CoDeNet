# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------


import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from ..functions.deform_conv import DeformConvFunction


class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConv, self).__init__()
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        assert out_channels % deformable_groups == 0, 'out_channels must be divisible by deformable groups'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = Parameter(torch.Tensor(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size).cuda())
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels).cuda())
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data, offset):
        return DeformConvFunction.apply(data, offset, self.weight, self.bias, self.in_channels, self.out_channels,
                                        self.kernel_size, self.stride, self.padding, self.dilation, self.groups,
                                        self.deformable_groups)


class DeformConvWithOffset(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConvWithOffset, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, kernel_size * kernel_size * 2 * deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

    def forward(self, x):
        return self.conv(x, self.conv_offset(x))


class DeformConvWithOffsetBound(nn.Module):
    
    def __init__(self, in_channels, out_channels, offset_bound=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConvWithOffsetBound, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, kernel_size * kernel_size * 2 * deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv_bound = torch.nn.Hardtanh(min_val=-offset_bound, max_val=offset_bound, inplace=True)
        self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

    def forward(self, x):
        return self.conv(x, self.conv_bound(self.conv_offset(x)))


# class DeformConvWithOffsetBoundPositive(nn.Module):
    
#     def __init__(self, in_channels, out_channels, offset_bound=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
#         super(DeformConvWithOffsetBoundPositive, self).__init__()
#         self.conv_offset = nn.Conv2d(in_channels, kernel_size * kernel_size * 2 * deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
#         self.conv_offset.weight.data.zero_()
#         self.conv_offset.bias.data.zero_()
#         self.conv_bound = torch.nn.Hardtanh(min_val=0, max_val=offset_bound, inplace=True)
#         self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                                padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

#     def forward(self, x):
#         return self.conv(x, self.conv_bound(self.conv_offset(x)))


class DeformConvWithOffsetBoundMinMax(nn.Module):
    
    def __init__(self, in_channels, out_channels, bound_min=4, bound_max=4):
        super(DeformConvWithOffsetBoundMinMax, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, 36, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.bound_min = torch.nn.Hardtanh(min_val=-bound_min, max_val=bound_min, inplace=True)
        self.conv_min = DeformConv(in_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1, dilation=1, groups=1, 
                                   deformable_groups=1, bias=False)
        # self.bound_max = torch.nn.Hardtanh(min_val=-bound_max, max_val=bound_max, inplace=True)
        # self.conv_max = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
        #                        padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)
        self.bound_max = torch.nn.Hardtanh(min_val=-bound_max, max_val=bound_max, inplace=True)
        self.conv_max = DeformConv(in_channels, out_channels, kernel_size=3, stride=1,
                                   padding=1+bound_min, dilation=1+bound_min, 
                                   groups=1, deformable_groups=1, bias=False)

    def forward(self, x):
        offset_min, offset_max = torch.chunk(self.conv_offset(x), 2, dim=1)
        return self.conv_min(x, self.bound_min(offset_min)) + self.conv_max(x, self.bound_max(offset_max))


class DeformConvWithOffsetRound(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConvWithOffsetRound, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, kernel_size * kernel_size * 2 * deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

    def forward(self, x):
        return self.conv(x, self.conv_offset(x).round_())


class DeformConvWithOffsetScale(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConvWithOffsetScale, self).__init__()
        self.conv_scale = nn.Conv2d(in_channels, deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_scale.weight.data.zero_()
        # self.conv_scale.bias.data.zero_()
        nn.init.constant_(self.conv_scale.bias.data, 1)
        self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

        self.anchor_offset = torch.FloatTensor([-1, -1, -1, 0, -1, 1, 
                                                 0, -1,  0, 0,  0, 1, 
                                                 1, -1,  1, 0,  1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    def forward(self, x):
        o = self.anchor_offset.to(x.device) * (self.conv_scale(x) - 1)
        return self.conv(x, o)


class DeformConvWithOffsetScaleGauss(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConvWithOffsetScaleGauss, self).__init__()
        self.conv_scale = nn.Conv2d(in_channels, deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_scale.weight.data.zero_()
        # self.conv_scale.bias.data.zero_()
        nn.init.constant_(self.conv_scale.bias.data, 1)
        self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

        self.anchor_default = torch.FloatTensor([-1, -1, -1, 0, -1, 1, 
                                                 0, -1,  0, 0,  0, 1, 
                                                 1, -1,  1, 0,  1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        self.anchor_gauss = torch.FloatTensor([-0.7071, -0.7071, -1, 0, -0.7071, 0.7071,
                                                 0, -1,  0, 0,  0, 1,
                                                0.7071, -0.7071,  1, 0,  0.7071, 0.7071]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    def forward(self, x):
        o =  self.conv_scale(x) * self.anchor_gauss.to(x.device) - self.anchor_default.to(x.device)
        return self.conv(x, o)


class DeformConvWithOffsetScaleGaussBound(nn.Module):
    
    def __init__(self, in_channels, out_channels, offset_bound=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConvWithOffsetScaleGaussBound, self).__init__()
        self.conv_scale = nn.Conv2d(in_channels, deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_scale.weight.data.zero_()
        # self.conv_scale.bias.data.zero_()
        nn.init.constant_(self.conv_scale.bias.data, 1)
        self.conv_bound = torch.nn.Hardtanh(min_val=-offset_bound, max_val=offset_bound, inplace=True)
        self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

        self.anchor_default = torch.FloatTensor([-1, -1, -1, 0, -1, 1, 
                                                 0, -1,  0, 0,  0, 1, 
                                                 1, -1,  1, 0,  1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        self.anchor_gauss = torch.FloatTensor([-0.7071, -0.7071, -1, 0, -0.7071, 0.7071,
                                                     0,      -1,  0, 0,       0,      1,
                                                0.7071, -0.7071,  1, 0,  0.7071, 0.7071]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    def forward(self, x):
        s = self.conv_bound(self.conv_scale(x))
        o = s * self.anchor_gauss.to(x.device) - self.anchor_default.to(x.device)
        return self.conv(x, o)


class DeformConvWithOffsetScaleGaussBoundPositive(nn.Module):
    
    def __init__(self, in_channels, out_channels, offset_bound=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConvWithOffsetScaleGaussBoundPositive, self).__init__()
        self.conv_scale = nn.Conv2d(in_channels, deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_scale.weight.data.zero_()
        # self.conv_scale.bias.data.zero_()
        nn.init.constant_(self.conv_scale.bias.data, 2)
        self.conv_bound = torch.nn.Hardtanh(min_val=0, max_val=offset_bound, inplace=True)
        # self.conv_bound = torch.nn.Hardtanh(min_val=1.5, max_val=offset_bound, inplace=True)
        self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

        self.anchor_default = torch.FloatTensor([-1, -1, -1, 0, -1, 1, 
                                                 0, -1,  0, 0,  0, 1, 
                                                 1, -1,  1, 0,  1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        self.anchor_gauss = torch.FloatTensor([-0.7071, -0.7071, -1, 0, -0.7071, 0.7071,
                                                     0,      -1,  0, 0,       0,      1,
                                                0.7071, -0.7071,  1, 0,  0.7071, 0.7071]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    def forward(self, x):
        s = self.conv_bound(self.conv_scale(x))
        o = s * self.anchor_gauss.to(x.device) - self.anchor_default.to(x.device)
        return self.conv(x, o)


class DeformConvWithOffsetScaleGaussBoundMinMax(nn.Module):
    
    def __init__(self, in_channels, out_channels, bound_min=8, bound_max=16):
        super(DeformConvWithOffsetScaleGaussBoundMinMax, self).__init__()
        self.conv_scale = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_scale.weight.data.zero_()
        # self.conv_scale.bias.data.zero_()
        nn.init.constant_(self.conv_scale.bias.data, 1)
        self.bound_min = torch.nn.Hardtanh(min_val=0, max_val=bound_min, inplace=True)
        self.conv_min = DeformConv(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, dilation=1, groups=1, deformable_groups=1, bias=True)

        self.bound_max = torch.nn.Hardtanh(min_val=bound_min, max_val=bound_max, inplace=True)
        self.conv_max = DeformConv(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, dilation=1, groups=1, deformable_groups=1, bias=True)

        self.anchor_default = torch.FloatTensor([-1, -1, -1, 0, -1, 1, 
                                                 0, -1,  0, 0,  0, 1, 
                                                 1, -1,  1, 0,  1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        self.anchor_gauss = torch.FloatTensor([-0.7071, -0.7071, -1, 0, -0.7071, 0.7071,
                                                     0,      -1,  0, 0,       0,      1,
                                                0.7071, -0.7071,  1, 0,  0.7071, 0.7071]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    def forward(self, x):
        s_min, s_max = torch.chunk(self.conv_scale(x), 2, dim=1)
        o_min, o_max = self.bound_min(s_min) * self.anchor_gauss.to(x.device) - self.anchor_default.to(x.device), \
                       self.bound_max(s_max) * self.anchor_gauss.to(x.device) - self.anchor_default.to(x.device)
        return self.conv_min(x, o_min) + self.conv_max(x, o_max)


class DeformConvWithOffsetScaleGaussBoundMinMaxShared(nn.Module):
    
    def __init__(self, in_channels, out_channels, bound_min=8, bound_max=16, 
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=True):
        super(DeformConvWithOffsetScaleGaussBoundMinMaxShared, self).__init__()
        
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        assert out_channels % deformable_groups == 0, 'out_channels must be divisible by deformable groups'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = Parameter(torch.Tensor(
            self.out_channels, self.in_channels // self.groups, *self.kernel_size).cuda())
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels).cuda())
        else:
            self.register_parameter('bias', None)

        stdv = 1. / math.sqrt(self.in_channels * kernel_size * kernel_size)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        self.anchor_default = torch.FloatTensor([-1, -1, -1, 0, -1, 1, 
                                                 0, -1,  0, 0,  0, 1, 
                                                 1, -1,  1, 0,  1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        self.anchor_gauss = torch.FloatTensor([-0.7071, -0.7071, -1, 0, -0.7071, 0.7071,
                                                     0,      -1,  0, 0,       0,      1,
                                                0.7071, -0.7071,  1, 0,  0.7071, 0.7071]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

        self.conv_scale = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_scale.weight.data.zero_()
        # self.conv_scale.bias.data.zero_()
        nn.init.constant_(self.conv_scale.bias.data, 1)
        self.bmin = torch.nn.Hardtanh(min_val=0, max_val=bound_min, inplace=True)
        self.bmax = torch.nn.Hardtanh(min_val=bound_min, max_val=bound_max, inplace=True)

    def forward(self, x):
        s_min, s_max = torch.chunk(self.conv_scale(x), 2, dim=1)
        o_min, o_max = self.bmin(s_min) * self.anchor_gauss.to(x.device) - self.anchor_default.to(x.device), \
                       self.bmax(s_max) * self.anchor_gauss.to(x.device) - self.anchor_default.to(x.device)
        return DeformConvFunction.apply(x, o_min, self.weight, self.bias, self.in_channels, self.out_channels,
                                        self.kernel_size, self.stride, self.padding, self.dilation, self.groups,
                                        self.deformable_groups) + \
               DeformConvFunction.apply(x, o_max, self.weight, self.bias, self.in_channels, self.out_channels,
                                        self.kernel_size, self.stride, self.padding, self.dilation, self.groups,
                                        self.deformable_groups)


class DeformConvWithOffsetScaleBound(nn.Module):
    
    def __init__(self, in_channels, out_channels, offset_bound=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConvWithOffsetScaleBound, self).__init__()
        self.conv_scale = nn.Conv2d(in_channels, deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_scale.weight.data.zero_()
        # self.conv_scale.bias.data.zero_()
        nn.init.constant_(self.conv_scale.bias.data, 1)
        self.conv_bound = torch.nn.Hardtanh(min_val=-offset_bound, max_val=offset_bound, inplace=True)
        self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

        self.anchor_offset = torch.FloatTensor([-1, -1, -1, 0, -1, 1, 
                                                 0, -1,  0, 0,  0, 1, 
                                                 1, -1,  1, 0,  1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    def forward(self, x):
        s = self.conv_bound(self.conv_scale(x))
        o = self.anchor_offset.to(x.device) * (s - 1)
        return self.conv(x, o)


class DeformConvWithOffsetScaleBoundPositive(nn.Module):
    
    def __init__(self, in_channels, out_channels, offset_bound=8, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, deformable_groups=1, bias=False):
        super(DeformConvWithOffsetScaleBoundPositive, self).__init__()
        self.conv_scale = nn.Conv2d(in_channels, deformable_groups, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_scale.weight.data.zero_()
        # self.conv_scale.bias.data.zero_()
        nn.init.constant_(self.conv_scale.bias.data, 1)
        self.conv_bound = torch.nn.Hardtanh(min_val=0, max_val=offset_bound, inplace=True)
        self.conv = DeformConv(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=groups, deformable_groups=deformable_groups, bias=bias)

        self.anchor_offset = torch.FloatTensor([-1, -1, -1, 0, -1, 1, 
                                                 0, -1,  0, 0,  0, 1, 
                                                 1, -1,  1, 0,  1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    def forward(self, x):
        s = self.conv_bound(self.conv_scale(x))
        o = self.anchor_offset.to(x.device) * (s - 1)
        return self.conv(x, o)


class DeformConvWithOffsetScaleBoundMinMax(nn.Module):
    
    def __init__(self, in_channels, out_channels, bound_min=4, bound_max=8):
        super(DeformConvWithOffsetScaleBoundMinMax, self).__init__()
        self.conv_scale = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_scale.weight.data.zero_()
        # self.conv_scale.bias.data.zero_()
        nn.init.constant_(self.conv_scale.bias.data, 1)
        self.bound_min = torch.nn.Hardtanh(min_val=0, max_val=bound_min, inplace=True)
        self.conv_min = DeformConv(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, dilation=1, groups=1, deformable_groups=1, bias=True)

        self.bound_max = torch.nn.Hardtanh(min_val=bound_min, max_val=bound_max, inplace=True)
        self.conv_max = DeformConv(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, dilation=1, groups=1, deformable_groups=1, bias=True)

        self.anchor_offset = torch.FloatTensor([-1, -1, -1, 0, -1, 1, 
                                                 0, -1,  0, 0,  0, 1, 
                                                 1, -1,  1, 0,  1, 1]).unsqueeze(0).unsqueeze(2).unsqueeze(2)

    def forward(self, x):
        s_min, s_max = torch.chunk(self.conv_scale(x), 2, dim=1)
        o_min, o_max = self.anchor_offset.to(x.device) * (self.bound_min(s_min) - 1), \
                       self.anchor_offset.to(x.device) * (self.bound_max(s_max) - 1)
        return self.conv_min(x, o_min) + self.conv_max(x, o_max)