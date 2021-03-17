import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn.modules.conv import Conv2d as _Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d as _BN2d
from pytorchcv.models.shufflenetv2 import ShuffleUnit as _SflUnit
from pytorchcv.models.common import ChannelShuffle
from torch.nn import Linear as _linear
from torch.nn import Embedding as _Embedding
from torch.nn import Module, Parameter
from .quantization_utils.quant_utils import *
import sys
sys.path.append('../')
from lib.models.external.functions.dcn_deform_conv import deform_conv
from lib.models.networks.shufflenetv2_dcn import channel_shuffle


## Basic Quantization Modules
class QuantLinear(_linear):
    """
    Quantized Module for Linear Layer

    """
    def __init__(self,
                 weight_bit,
                 input_size,
                 output_size,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 alpha=None,
                 per_channel=True,
                 group_quantization=False,
                 group_number=60,
                 weight_percentile=False):
        super(QuantLinear, self).__init__(input_size, output_size)
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.alpha = alpha
        self.quant_mode = quant_mode
        self.input_size = input_size
        self.output_size = output_size
        self.momentum = 0.99

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))

        self.per_channel = per_channel
        self.weight_percentile = weight_percentile

        self.group_quantization = group_quantization
        self.group_number = group_number

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def reset_bits(self, weight_bit=8):
        self.full_precision_flag = False
        self.weight_bit = weight_bit

    def reset_alpha(self, alpha):
        assert alpha >= 0.0
        assert alpha <= 1.0
        self.alpha = alpha

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s += "\n--> quantized to weight_bit={}, full_precision_flag={}".format(self.weight_bit, self.full_precision_flag)
        return s

    def forward(self, x):
        w = self.weight
        self.channel_num = w.shape[1]

        if self.per_channel:
            if not self.group_quantization:
                x_transform = w.data.transpose(0, 1).contiguous()
                w_min = x_transform.min(dim=1)[0]
                w_max = x_transform.max(dim=1)[0]

                if not self.weight_percentile:
                    pass

                elif self.weight_percentile:
                    lower_percentile = 0.1
                    upper_percentile = 99.9
                    input_length = x_transform[0].view(-1).shape[0]

                    lower_index = round(input_length * lower_percentile * 0.01)
                    upper_index = round(input_length * upper_percentile * 0.01)

                    lower_bound, _ = torch.topk(x_transform, lower_index, largest=False, sorted=False)
                    upper_bound, _ = torch.topk(x_transform, input_length - upper_index, largest=True, sorted=False)

                    w_min = lower_bound.max(dim=1)[0]
                    w_max = upper_bound.min(dim=1)[0]

            elif self.group_quantization:
                x_transform = w.data.transpose(0, 1).contiguous()
                w_min = x_transform.min(dim=1)[0]
                w_max = x_transform.max(dim=1)[0]

                # group_length should be an integer
                group_length = w_max.size()[0] // self.group_number

                if not self.weight_percentile:
                    temp_w_min = w_min.clone()
                    temp_w_max = w_max.clone()

                    for i in range(self.group_number):
                        w_min[i * group_length: (i + 1) * group_length] = \
                            temp_w_min[i * group_length: (i + 1) * group_length].min().repeat(group_length)
                        w_max[i * group_length: (i + 1) * group_length] = \
                            temp_w_max[i * group_length: (i + 1) * group_length].max().repeat(group_length)

                elif self.weight_percentile:
                    for i in range(self.group_number):
                        temp_w_min, temp_w_max = get_percentile_min_max(x_transform[i * group_length: (i + 1) * group_length].view(-1), 0.1, 99.9, output_tensor=True)
                        w_min[i * group_length: (i + 1) * group_length] = temp_w_min.repeat(group_length)
                        w_max[i * group_length: (i + 1) * group_length] = temp_w_max.repeat(group_length)

        elif not self.per_channel:
            if not self.weight_percentile:
                w_min = w.data.min().expand(1)
                w_max = w.data.max().expand(1)
            elif self.weight_percentile:
                w_min, w_max = get_percentile_min_max(w.view(-1), 0.1, 99.9, output_tensor=True)

        if self.x_min.size()[0] == 1:
            if self.x_min == self.x_max:
                self.x_min = w_min
                self.x_max = w_max

        # exponential moving average (EMA)
        # use momentum to prevent the quantized values from changing greatly every iteration
        self.x_min = self.momentum * self.x_min + (1. - self.momentum) * w_min
        self.x_max = self.momentum * self.x_max + (1. - self.momentum) * w_max

        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, self.x_min,
                                     self.x_max, self.per_channel, self.weight_percentile)
        else:
            w = self.weight

        if self.alpha is None:
            return F.linear(x, w, bias=self.bias)
        else:
            assert self.full_precision_flag == False

            quantized = self.alpha * w
            non_quantized = (1 - self.alpha) * self.weight

            return F.linear(x, quantized + non_quantized, bias=self.bias)


class QuantAct(Module):
    """
    Quantized Module for Activation Layer

    """
    def __init__(self,
                 activation_bit,
                 momentum=0.99,
                 full_precision_flag=False,
                 running_stat=True,
                 quant_mode="symmetric",
                 show_flag=False,
                 percentile=False):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.momentum = momentum
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.show_flag = show_flag
        self.percentile = percentile

        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))

        if quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.act_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "full_precision_flag={2}, Act_min: {3:.2f}, " \
               "Act_max: {4:.2f})".format(self.__class__.__name__, self.activation_bit,
                                     self.full_precision_flag, self.x_min.item(), self.x_max.item())

    def forward(self, x):
        if self.running_stat:
            if not self.percentile:
                x_min = x.data.min()
                x_max = x.data.max()
            else:
                x_min, x_max = get_percentile_min_max(x.detach().view(-1), 0.1, 99.9, output_tensor=True)

            # Initialization
            if self.x_min == self.x_max:
                self.x_min += x_min
                self.x_max += x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values from changing greatly every iteration
            else:
                self.x_min += (self.momentum - 1.) * self.x_min + (1. - self.momentum) * x_min
                self.x_max += (self.momentum - 1.) * self.x_max + (1. - self.momentum) * x_max

        if not self.full_precision_flag:
            quant_act = self.act_function(x, self.activation_bit, self.x_min, self.x_max)
            return quant_act
        else:
            return x


class Quant_Conv2d(Module):
    """
    Quantized Module for Convolution Layer

    """
    def __init__(self,
                 weight_bit,
                 bias_bit=None, 
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=False,
                 weight_percentile=False):
        super(Quant_Conv2d, self).__init__()

        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.momentum = 0.99
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def __repr__(self):
        s = super(Quant_Conv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        w = self.weight

        if self.per_channel:
            x_transform = w.data.contiguous().view(self.out_channels, -1)

            if not self.weight_percentile:
                w_min = x_transform.min(dim=1).values
                w_max = x_transform.max(dim=1).values
            elif self.weight_percentile:
                lower_percentile = 0.1
                upper_percentile = 99.9
                input_length = x_transform.shape[1]

                if input_length < 10:
                    w_min = x_transform.min(dim=1).values * 0.95
                    w_max = x_transform.max(dim=1).values * 0.95
                else:
                    lower_index = math.ceil(input_length * lower_percentile * 0.01)
                    upper_index = math.ceil(input_length * upper_percentile * 0.01)

                    w_min = torch.kthvalue(x_transform, k=lower_index, dim=1).values
                    w_max = torch.kthvalue(x_transform, k=upper_index, dim=1).values
            if self.quantize_bias:
                raise NotImplementedError('channel-wise quantize bias is not supported')

        elif not self.per_channel:
            if not self.weight_percentile:
                w_min = w.data.min()
                w_max = w.data.max()
            elif self.weight_percentile:
                w_min, w_max = get_percentile_min_max(w.view(-1), 0.1, 99.9, output_tensor=True)
            if self.quantize_bias:
                b_min = self.bias.min()
                b_max = self.bias.max()

        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max, self.per_channel, self.weight_percentile)
            if self.quantize_bias:
                b = self.weight_function(self.bias, self.bias_bit, b_min, b_max, self.per_channel, False)
        else:
            w = self.weight

        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QuantBnConv2d(Module):
    """
    Quantized Module for BN + Convolution (with BN Folding)

    """
    def __init__(self,
                 weight_bit,
                 bias_bit=None,
                 full_precision_flag=False,
                 running_stat=True,
                 quant_mode="asymmetric",
                 per_channel=False,
                 weight_percentile=False):
        super(QuantBnConv2d, self).__init__()
        self.weight_bit = weight_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def set_param(self, conv, bn):
        self.conv = conv
        self.bn = bn

    def __repr__(self):
        conv_s = super(QuantBnConv2d, self).__repr__()
        s= "({0}, weight_bit={1}, bias_bit={2}, groups={3}, wt-channel-wise={4}, wt-percentile={5}, " \
           "act-percentile={6})".format(conv_s, self.weight_bit, self.bias_bit, self.conv.groups,
                                        self.per_channel, self.weight_percentile, False)
        return s

    def forward(self, x):
        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = self.conv.weight * scale_factor.reshape([self.conv.out_channels, 1, 1, 1])
        if self.conv.bias is not None:
            scaled_bias = self.conv.bias
        else:
            scaled_bias = torch.zeros_like(self.bn.running_mean)
        scaled_bias = (scaled_bias - self.bn.running_mean) * scale_factor + self.bn.bias

        if not self.full_precision_flag:
            if self.per_channel:
                x_transform = scaled_weight.data.contiguous().view(self.conv.out_channels, -1)

                if not self.weight_percentile:
                    w_min = x_transform.min(dim=1).values
                    w_max = x_transform.max(dim=1).values

                elif self.weight_percentile:
                    lower_percentile = 0.1
                    upper_percentile = 99.9
                    input_length = x_transform.shape[1]

                    if input_length < 10:
                        w_min = x_transform.min(dim=1).values * 0.95
                        w_max = x_transform.max(dim=1).values * 0.95
                    else:
                        lower_index = math.ceil(input_length * lower_percentile * 0.01)
                        upper_index = math.ceil(input_length * upper_percentile * 0.01)

                        w_min = torch.kthvalue(x_transform, k=lower_index, dim=1).values
                        w_max = torch.kthvalue(x_transform, k=upper_index, dim=1).values

                if self.quantize_bias:
                    raise NotImplementedError

            elif not self.per_channel:
                if not self.weight_percentile:
                    w_min = scaled_weight.data.min()
                    w_max = scaled_weight.data.max()
                elif self.weight_percentile:
                    w_min, w_max = get_percentile_min_max(scaled_weight.view(-1),
                                        lower_percentile, upper_percentile, output_tensor=True)

                if self.quantize_bias:
                    b_min = scaled_bias.data.min()
                    b_max = scaled_bias.data.max()

            scaled_weight = self.weight_function(scaled_weight, self.weight_bit,
                                        w_min, w_max, self.per_channel, self.weight_percentile)
            if self.quantize_bias:
                scaled_bias = self.weight_function(scaled_bias, self.bias_bit,
                                        b_min, b_max, self.per_channel, False)

        return F.conv2d(x, scaled_weight, scaled_bias, self.conv.stride,
                        self.conv.padding, self.conv.dilation, self.conv.groups)


class QuantDeformConv2d(Module):
    """
    Quantized Module for Deformable Convolution Layer

    """
    def __init__(self,
                 weight_bit,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=False,
                 weight_percentile=False):
        super(QuantDeformConv2d, self).__init__()

        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.momentum = 0.99
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def __repr__(self):
        s = super(QuantDeformConv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, full_precision_flag={})".format(self.weight_bit, self.full_precision_flag)
        return s

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.deformable_groups = conv.deformable_groups

        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
        except AttributeError:
            self.bias = None

    def forward(self, x, offset):
        w = self.weight

        if self.per_channel:
            x_transform = w.data.contiguous().view(self.out_channels, -1)

            if not self.weight_percentile:
                w_min = x_transform.min(dim=1).values
                w_max = x_transform.max(dim=1).values
            elif self.weight_percentile:
                lower_percentile = 0.1
                upper_percentile = 99.9
                input_length = x_transform.shape[1]

                if input_length < 10:
                    w_min = x_transform.min(dim=1).values * 0.95
                    w_max = x_transform.max(dim=1).values * 0.95
                else:
                    lower_index = math.ceil(input_length * lower_percentile * 0.01)
                    upper_index = math.ceil(input_length * upper_percentile * 0.01)

                    w_min = torch.kthvalue(x_transform, k=lower_index, dim=1).values
                    w_max = torch.kthvalue(x_transform, k=upper_index, dim=1).values
            if self.quantize_bias:
                raise NotImplementedError('channel-wise quantize bias is not supported')

        elif not self.per_channel:
            if not self.weight_percentile:
                w_min = w.data.min()
                w_max = w.data.max()
            elif self.weight_percentile:
                w_min, w_max = get_percentile_min_max(w.view(-1), 0.1, 99.9, output_tensor=True)
            if self.quantize_bias:
                b_min = self.bias.min()
                b_max = self.bias.max()

        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max, self.per_channel,
                                     self.weight_percentile)
            if self.quantize_bias:
                b = self.weight_function(self.bias, self.bias_bit, b_min, b_max, self.per_channel, False)
        else:
            w = self.weight

        return deform_conv(x, offset, w, self.stride, self.padding, self.dilation, self.groups, self.deformable_groups)


class QuantBnDeformConv2d(Module):
    """
    Quantized Module for BN + Deformable Convolution (with BN Folding)

    """
    def __init__(self,
                 weight_bit,
                 bias_bit=None,
                 full_precision_flag=False,
                 running_stat=True,
                 quant_mode="symmetric",
                 per_channel=False,
                 weight_percentile=False):
        super(QuantBnDeformConv2d, self).__init__()
        self.weight_bit = weight_bit
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True

        if quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(quant_mode))

    def set_param(self, conv, bn):

        self.conv = conv
        self.bn = bn

    def __repr__(self):
        conv_s = super(QuantBnDeformConv2d, self).__repr__()
        s = "({0}, weight_bit={1}, bias_bit={2}, groups={3}, wt-channel-wise={4}, " \
            "wt-percentile={5}, act-percentile={6})".format(conv_s, self.weight_bit, self.bias_bit,
                                    self.conv.groups, self.per_channel, self.weight_percentile, False)
        return s

    def forward(self, x, offset):

        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = self.conv.weight * scale_factor.reshape([self.conv.out_channels, 1, 1, 1])
        if self.conv.bias is not None:
            scaled_bias = self.conv.bias
        else:
            scaled_bias = torch.zeros_like(self.bn.running_mean)
        scaled_bias = (scaled_bias - self.bn.running_mean) * scale_factor + self.bn.bias

        if not self.full_precision_flag:
            if self.per_channel:
                x_transform = scaled_weight.data.contiguous().view(self.conv.out_channels, -1)

                if not self.weight_percentile:
                    w_min = x_transform.min(dim=1).values
                    w_max = x_transform.max(dim=1).values

                elif self.weight_percentile:
                    lower_percentile = 0.1
                    upper_percentile = 99.9
                    input_length = x_transform.shape[1]

                    if input_length < 10:
                        w_min = x_transform.min(dim=1).values * 0.95
                        w_max = x_transform.max(dim=1).values * 0.95
                    else:
                        lower_index = math.ceil(input_length * lower_percentile * 0.01)
                        upper_index = math.ceil(input_length * upper_percentile * 0.01)

                        w_min = torch.kthvalue(x_transform, k=lower_index, dim=1).values
                        w_max = torch.kthvalue(x_transform, k=upper_index, dim=1).values

                if self.quantize_bias:
                    raise NotImplementedError

            elif not self.per_channel:
                if not self.weight_percentile:
                    w_min = scaled_weight.data.min()
                    w_max = scaled_weight.data.max()
                elif self.weight_percentile:
                    w_min, w_max = get_percentile_min_max(scaled_weight.view(-1), lower_percentile, upper_percentile,
                                                          output_tensor=True)

                if self.quantize_bias:
                    b_min = scaled_bias.data.min()
                    b_max = scaled_bias.data.max()

            scaled_weight = self.weight_function(scaled_weight, self.weight_bit, w_min, w_max, self.per_channel,
                                                 self.weight_percentile)
            if self.quantize_bias:
                scaled_bias = self.weight_function(scaled_bias, self.bias_bit, b_min, b_max, self.per_channel, False)

        output = deform_conv(x, offset, scaled_weight, self.conv.stride, self.conv.padding, self.conv.dilation,
                        self.conv.groups, self.conv.deformable_groups)
        return output + scaled_bias.view(1, -1, 1, 1).expand(output.size())


## First-class Compound Quantization Modules: complex quantization modules built on basic modules
class QuantDeformConvWithOffsetScaleBoundPositive(Module):
    """
    Quantized Counterpart of DeformConvWithOffsetScaleBoundPositive Module Used in CoDeNet

    """
    def __init__(self,
                weight_bit,
                act_bit,
                full_precision_flag=False,
                bias_bit=None,
                act_percentile=False,
                wt_quant_mode='symmetric',
                act_quant_mode='symmetric',
                per_channel=False,
                weight_percentile=False):
        super(QuantDeformConvWithOffsetScaleBoundPositive, self).__init__()
        self.act_bit = act_bit
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.wt_quant_mode = wt_quant_mode
        self.act_quant_mode = act_quant_mode
        self.full_precision_flag = full_precision_flag
        self.act_percentile = act_percentile
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile

    def set_param(self, deform_conv, bn):
        self.quant_conv_scale = Quant_Conv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                             per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_conv_scale.set_param(deform_conv.conv_scale)
        self.quant_act = nn.Sequential(*[deform_conv.conv_bound, QuantAct(self.act_bit,
                                                                          quant_mode="asymmetric",
                                                                          percentile=self.act_percentile)])

        self.quant_deform_conv = QuantDeformConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                                per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_deform_conv.set_param(deform_conv.conv)
        self.quant_identity_deform = QuantAct(self.act_bit,
                                              quant_mode=self.act_quant_mode, percentile=self.act_percentile)

        self.anchor_offset = deform_conv.anchor_offset.clone()

        self.quant_conv_channel_bn = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                                per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_conv_channel_bn.set_param(deform_conv.conv_channel, bn)

    def forward(self, x):
        s = self.quant_act(self.quant_conv_scale(x))
        o = self.anchor_offset.to(x.device) * (s - 1)
        return self.quant_conv_channel_bn(self.quant_identity_deform(self.quant_deform_conv(x, o)))


class QuantDeformConvWithOffsetScaleBoundPositiveBn(Module):
    """
    Quantized Counterpart of DeformConvWithOffsetScaleBoundPositive Module Used in CoDeNet with BN Folding

    """
    def __init__(self,
                weight_bit,
                act_bit,
                full_precision_flag=False,
                bias_bit=None,
                act_percentile=False,
                wt_quant_mode='symmetric',
                act_quant_mode='symmetric',
                per_channel=False,
                weight_percentile=False):
        super(QuantDeformConvWithOffsetScaleBoundPositiveBn, self).__init__()
        self.act_bit = act_bit
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.wt_quant_mode = wt_quant_mode
        self.act_quant_mode = act_quant_mode
        self.full_precision_flag = full_precision_flag
        self.act_percentile = act_percentile
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile

    def set_param(self, deform_conv, bn):
        self.quant_conv_scale = Quant_Conv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                             per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_conv_scale.set_param(deform_conv.conv_scale)
        self.quant_act = nn.Sequential(*[deform_conv.conv_bound, QuantAct(self.act_bit,
                                                                          quant_mode="asymmetric",
                                                                          percentile=self.act_percentile)])

        self.quant_deform_conv_bn = QuantBnDeformConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                                   per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_deform_conv_bn.set_param(deform_conv.conv, bn)

        self.anchor_offset = deform_conv.anchor_offset.clone()

    def forward(self, x):
        s = self.quant_act(self.quant_conv_scale(x))
        o = self.anchor_offset.to(x.device) * (s - 1)
        return self.quant_deform_conv_bn(x, o)


## Second-class Compound Quantization Modules:
## complex quantization modules built on basic/first-class quantiation modules
class QuantSflUnit(Module):
    """
    Quantized Sfl_Unit Defined in PyTorchCV
    Important Note: The implementation of sequenced Sfl_Units
    requires to use shared activation quantization at the residuel of each Sfl_Unit.

    """
    def __init__(self,
                weight_bit, 
                act_bit, 
                full_precision_flag=False,
                bias_bit=None,
                act_percentile=False,
                wt_quant_mode='symmetric',
                act_quant_mode='symmetric',
                per_channel=False):
        super(QuantSflUnit, self).__init__()
        self.act_bit = act_bit
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.wt_quant_mode = wt_quant_mode
        self.act_quant_mode = act_quant_mode
        self.full_precision_flag = full_precision_flag
        self.act_percentile = act_percentile
        self.per_channel = per_channel

    def set_param(self, sfl_unit):
        self.downsample = sfl_unit.downsample
        self.use_se = sfl_unit.use_se
        self.use_residual = sfl_unit.use_residual
        self.c_shuffle = ChannelShuffle(channels=2, groups=2)

        # there are 5/3 (w. downsample/w.o. downsample) convolutions in a sfl_unit
        self.quant_compr_convbn1 = QuantBnConv2d(weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                                                 quant_mode=self.wt_quant_mode, per_channel=self.per_channel)
        self.quant_compr_convbn1.set_param(sfl_unit.compress_conv1, sfl_unit.compress_bn1)
        self.quant_act1 = nn.Sequential(*[nn.ReLU(inplace=True), QuantAct(activation_bit=self.act_bit,
            quant_mode='asymmetric', full_precision_flag=self.full_precision_flag, percentile=self.act_percentile)])

        self.quant_dw_convbn2 = QuantBnConv2d(weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                                quant_mode=self.wt_quant_mode, weight_percentile=False, per_channel=self.per_channel)
        self.quant_dw_convbn2.set_param(sfl_unit.dw_conv2, sfl_unit.dw_bn2)
        self.quant_act2 = QuantAct(activation_bit=self.act_bit, quant_mode=self.act_quant_mode,
                                   full_precision_flag=self.full_precision_flag, percentile=self.act_percentile)

        self.quant_exp_convbn3 = QuantBnConv2d(weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                                               quant_mode=self.wt_quant_mode, per_channel=self.per_channel)
        self.quant_exp_convbn3.set_param(sfl_unit.expand_conv3, sfl_unit.expand_bn3)

        if self.downsample:
            self.quant_dw_convbn4 = QuantBnConv2d(weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                                                  quant_mode=self.wt_quant_mode, weight_percentile=False,
                                                  per_channel=self.per_channel)
            self.quant_dw_convbn4.set_param(sfl_unit.dw_conv4, sfl_unit.dw_bn4)
            self.quant_act4 = QuantAct(activation_bit=self.act_bit, quant_mode=self.act_quant_mode,
                                       full_precision_flag=self.full_precision_flag, percentile=self.act_percentile)

            self.quant_exp_convbn5 = QuantBnConv2d(weight_bit=self.weight_bit, bias_bit=self.bias_bit,
                                                   quant_mode=self.wt_quant_mode, per_channel=self.per_channel)
            self.quant_exp_convbn5.set_param(sfl_unit.expand_conv5, sfl_unit.expand_bn5)

    def set_act(self, share_quant_act):
        # set the last activation quantization to be the given shared module
        self.quant_act = share_quant_act

    def forward(self, x):
        if self.downsample:
            y1 = self.quant_dw_convbn4(x)
            y1 = self.quant_act4(y1)
            y1 = self.quant_exp_convbn5(y1)
            y1 = self.quant_act(y1)
            x2 = x
        else:
            y1, x2 = torch.chunk(x, chunks=2, dim=1)
        y2 = self.quant_compr_convbn1(x2)
        y2 = self.quant_act1(y2)
        y2 = self.quant_dw_convbn2(y2)
        y2 = self.quant_act2(y2)
        y2 = self.quant_exp_convbn3(y2)
        y2 = self.quant_act(y2)
        x = torch.cat((y1, y2), dim=1)
        x = self.c_shuffle(x)
        return x


class QuantBaseNode(Module):
    """
    Quantized BaseNode Defined in DCN
    Important Note: The implementation of sequenced BaseNodes
    requires to use shared activation quantization at the residuel of each BaseNode.

    """
    def __init__(self, 
                weight_bit, 
                act_bit,
                full_precision_flag=False,
                bias_bit=None,
                act_percentile=False,
                wt_quant_mode='symmetric',
                act_quant_mode='symmetric',
                per_channel=False,
                weight_percentile=False):
        # quantization settings
        super(QuantBaseNode, self).__init__()
        self.act_bit = act_bit
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.wt_quant_mode = wt_quant_mode
        self.act_quant_mode = act_quant_mode
        self.full_precision_flag = full_precision_flag
        self.act_percentile = act_percentile
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile

    def set_param(self, base_node):
        self.stride = base_node.stride

        conv1, bn1, act1 = base_node.b2[0], base_node.b2[1], base_node.b2[2]
        self.quant_convbn1 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                           per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_convbn1.set_param(conv1, bn1)
        self.quant_act1 = QuantAct(self.act_bit, quant_mode="asymmetric", percentile=self.act_percentile)

        conv2, bn2 = base_node.b2[3], base_node.b2[4]
        assert type(conv2) == nn.Conv2d
        self.quant_convbn2 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                           per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_convbn2.set_param(conv2, bn2)
        self.quant_act2 = QuantAct(self.act_bit, quant_mode=self.act_quant_mode, percentile=self.act_percentile)

        conv3, bn3, act3 = base_node.b2[5], base_node.b2[6], base_node.b2[7]
        self.quant_convbn3 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                           per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_convbn3.set_param(conv3, bn3)

        if base_node.stride == 2:
            conv4, bn4 = base_node.b1[0], base_node.b1[1]
            assert type(conv4) == nn.Conv2d
            self.quant_convbn4 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                               per_channel=self.per_channel, weight_percentile=self.weight_percentile)
            self.quant_convbn4.set_param(conv4, bn4)
            self.quant_act4 = QuantAct(self.act_bit, quant_mode=self.act_quant_mode, percentile=self.act_percentile)

            conv5, bn5, act5 = base_node.b1[2], base_node.b1[3], base_node.b1[4]
            self.quant_convbn5 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                               per_channel=self.per_channel, weight_percentile=self.weight_percentile)
            self.quant_convbn5.set_param(conv5, bn5)


    def set_act(self, share_quant_act):
        # set the last activation quantization to be the given shared module
        self.quant_act = share_quant_act

    def forward(self, x):
        # forward using the quantized modules
        if self.stride == 1:
            split = x.shape[1]//2
            x1 = x[:, :split, :, :]
            x2 = x[:, split:, :, :]

        else:
            x1 = self.quant_convbn4(x)
            x1 = self.quant_act4(x1)

            x1 = self.quant_convbn5(x1)
            x1 = nn.ReLU()(x1)
            x1 = self.quant_act(x1)
            x2 = x

        x2 = self.quant_convbn1(x2)
        x2 = nn.ReLU()(x2)
        x2 = self.quant_act1(x2)

        x2 = self.quant_convbn2(x2)
        x2 = self.quant_act2(x2)

        x2 = self.quant_convbn3(x2)
        x2 = nn.ReLU()(x2)
        x2 = self.quant_act(x2)

        y = torch.cat((x1, x2), dim=1)
        y = channel_shuffle(y, 2)
        return y


class QuantBaseNodeDeform(Module):
    """
    Quantized BaseNode (with Deformable Convolution) Defined in DCN
    Important Note: The implementation of sequenced BaseNodes
    requires to use shared activation quantization at the residuel of each BaseNode.

    """
    def __init__(self,
                weight_bit,
                act_bit,
                full_precision_flag=False,
                bias_bit=None,
                act_percentile=False,
                wt_quant_mode='symmetric',
                act_quant_mode='symmetric',
                per_channel=False,
                weight_percentile=False):
        super(QuantBaseNodeDeform, self).__init__()
        self.act_bit = act_bit
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.wt_quant_mode = wt_quant_mode
        self.act_quant_mode = act_quant_mode
        self.full_precision_flag = full_precision_flag
        self.act_percentile = act_percentile
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile

    def set_param(self, base_node):
        self.stride = base_node.stride

        conv1, bn1, act1 = base_node.b2[0], base_node.b2[1], base_node.b2[2]
        self.quant_convbn1 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                           per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_convbn1.set_param(conv1, bn1)
        self.quant_act1 = QuantAct(self.act_bit, quant_mode="asymmetric", percentile=self.act_percentile)

        conv2, bn2 = base_node.b2[3], base_node.b2[4]
        self.quant_convbn2 = QuantDeformConvWithOffsetScaleBoundPositiveBn(self.weight_bit, self.act_bit,
                                            act_percentile=self.act_percentile, wt_quant_mode=self.wt_quant_mode,
                                            act_quant_mode=act_quant_mode, per_channel=self.per_channel,
                                            weight_percentile=self.weight_percentile)
        self.quant_convbn2.set_param(conv2, bn2)
        self.quant_act2 = QuantAct(self.act_bit, quant_mode=self.act_quant_mode, percentile=self.act_percentile)

        conv3, bn3, act3 = base_node.b2[5], base_node.b2[6], base_node.b2[7]
        self.quant_convbn3 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                           per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_convbn3.set_param(conv3, bn3)

        if base_node.stride == 2:
            conv4, bn4 = base_node.b1[0], base_node.b1[1]
            self.quant_convbn4 = QuantDeformConvWithOffsetScaleBoundPositiveBn(self.weight_bit, self.act_bit,
                                            act_percentile=self.act_percentile, wt_quant_mode=self.wt_quant_mode,
                                            act_quant_mode=self.act_quant_mode, per_channel=self.per_channel,
                                            weight_percentile=self.weight_percentile)
            self.quant_convbn4.set_param(conv4, bn4)
            self.quant_act4 = QuantAct(self.act_bit, quant_mode=self.act_quant_mode, percentile=self.act_percentile)

            conv5, bn5, act5 = base_node.b1[2], base_node.b1[3], base_node.b1[4]
            self.quant_convbn5 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                               per_channel=self.per_channel, weight_percentile=self.weight_percentile)
            self.quant_convbn5.set_param(conv5, bn5)


    def set_act(self, share_quant_act):
        # set the last activation quantization to be the given shared module
        self.quant_act = share_quant_act

    def forward(self, x):
        # forward using the quantized modules
        from models.networks.sfl_dcn import channel_shuffle
        if self.stride == 1:
            split = x.shape[1]//2
            x1 = x[:, :split, :, :]
            x2 = x[:, split:, :, :]

        else:
            x1 = self.quant_convbn4(x)
            x1 = self.quant_act4(x1)

            x1 = self.quant_convbn5(x1)
            x1 = nn.ReLU()(x1)
            x1 = self.quant_act(x1)
            x2 = x

        x2 = self.quant_convbn1(x2)
        x2 = nn.ReLU()(x2)
        x2 = self.quant_act1(x2)

        x2 = self.quant_convbn2(x2)
        x2 = self.quant_act2(x2)

        x2 = self.quant_convbn3(x2)
        x2 = nn.ReLU()(x2)
        x2 = self.quant_act(x2)

        y = torch.cat((x1, x2), dim=1)
        y = channel_shuffle(y, 2)
        return y


class QuantDepthwiseNode(Module):
    """
    Quantized DepthwiseNode Defined in DCN

    """
    def __init__(self,
                weight_bit,
                act_bit,
                full_precision_flag=False,
                bias_bit=None,
                act_percentile=False,
                wt_quant_mode='symmetric',
                act_quant_mode='symmetric',
                per_channel=False,
                weight_percentile=False):
        super(QuantDepthwiseNode, self).__init__()
        self.act_bit = act_bit
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.wt_quant_mode = wt_quant_mode
        self.act_quant_mode = act_quant_mode
        self.full_precision_flag = full_precision_flag
        self.act_percentile = act_percentile
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile

    def set_param(self, head_node):
        conv1, bn1, act1 = head_node[0], head_node[1], head_node[2]
        self.quant_convbn1 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                           per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_convbn1.set_param(conv1, bn1)
        self.quant_act1 = nn.Sequential(*[act1, QuantAct(self.act_bit, quant_mode="asymmetric", percentile=self.act_percentile)])

        conv2, bn2 = head_node[3], head_node[4]
        assert type(conv2) == nn.Conv2d
        self.quant_convbn2 = QuantBnConv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                           per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_convbn2.set_param(conv2, bn2)

        act3 = head_node[5]
        self.quant_act3 = nn.Sequential(*[act3, QuantAct(self.act_bit, quant_mode="asymmetric", percentile=self.act_percentile)])

        conv = head_node[6]
        self.quant_conv = Quant_Conv2d(self.weight_bit, quant_mode=self.wt_quant_mode,
                                       per_channel=self.per_channel, weight_percentile=self.weight_percentile)
        self.quant_conv.set_param(conv)

    def forward(self, x):
        # forward using the quantized modules
        x = self.quant_convbn1(x)
        x = self.quant_act1(x)

        x = self.quant_convbn2(x)
        x = self.quant_act3(x)

        x = self.quant_conv(x)

        return x
