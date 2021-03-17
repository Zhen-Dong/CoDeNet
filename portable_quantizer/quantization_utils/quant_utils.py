import math
import numpy as np
from torch.autograd import Function, Variable
import torch

TensorT = torch.Tensor


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def get_percentile_min_max(input, lower_percentile, upper_percentile, output_tensor=False):
    input_length = input.shape[0]

    lower_index = round(input_length * lower_percentile * 0.01)
    upper_index = round(input_length * upper_percentile * 0.01)

    lower_bound = torch.kthvalue(input, k=lower_index).values
    upper_bound = torch.kthvalue(input, k=upper_index).values

    if not output_tensor:
        lower_bound = lower_bound.item()
        upper_bound = upper_bound.item()
    return lower_bound, upper_bound


def linear_quantize(input, scale, zero_point, inplace=False):
    scale = scale.view(-1, 1, 1, 1)
    zero_point = zero_point.view(-1, 1, 1, 1)
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    # scale and zero_point can be broadcast to the same shape as input
    # the * and - here are element-wise operations
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    scale = scale.view(-1, 1, 1, 1)
    zero_point = zero_point.view(-1, 1, 1, 1)
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    # scale and zero_point can be broadcast to the same shape as input
    # the + and / here are element-wise operations
    return (input + zero_point) / scale


def linear_quantize_clamp(input, scale, zero_point, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale, zero_point, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def asymmetric_linear_quantization_params(num_bits, saturation_min,
                                          saturation_max, integral_zero_point=True, signed=True):
    n = 2 ** num_bits - 1

    scale = n / torch.clamp((saturation_max - saturation_min), min=0.0000000001)

    zero_point = scale * saturation_min

    if integral_zero_point:
        if isinstance(zero_point, torch.Tensor):
            zero_point = zero_point.round()
        else:
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2 ** (num_bits - 1)
    return scale, zero_point


def symmetric_linear_quantization_params(num_bits, saturation_magnitude, signed=False):
    n = 2 ** (num_bits - 1) - 1
    scale = n / torch.clamp(saturation_magnitude, min=0.0000000001)
    zero_point = torch.zeros_like(scale)
    if signed:
        raise NotImplementedError
    return scale, zero_point


def affine_quant_func(x, k, lower_bound, upper_bound):
    """ Quantize input variables via affine methods.

    input type: TensorT, int, float, float
    output type: float, TensorT, int

    Returns:
            - delta: magnitude of quantized values;
            - quant_idx: same shape with x;
            - shift_idx: quantized value w.r.t. real value 0.

    """
    assert lower_bound <= upper_bound, "got lower_bound = {}, while upper_bound = {}".format(
        lower_bound, upper_bound)

    # asymmetic quantization, 2 ** k - 1 rather than 2 ** (k-1) - 1
    delta = (upper_bound - lower_bound) / (2. ** k - 1.)
    x_new = torch.clamp(x, lower_bound, upper_bound)

    quant_idx = torch.round((x_new - lower_bound) / delta)
    shift_idx = math.floor(abs(lower_bound) / delta)

    return delta, quant_idx, shift_idx


def nudge_min_max(k, x_min, x_max):
    """
    This function applies a small shift on data range to make sure 0 is quantized to exact 0.

    k is int type, x_min and x_max are float type.
    0 is important since there are lots of 0 in data, and it doesn't require operations.

    """
    assert x_min <= x_max, "got x_min = {}, while x_max = {}".format(x_min, x_max)

    modified_min, modified_max = x_min.clone(), x_max.clone()

    if 0. <= x_min:
        modified_min.zero_()
    elif x_max <= 0.:
        modified_max.zero_()
    else:
        modified_range = modified_max - modified_min
        delta = modified_range / (2. ** k - 1.)
        mismatch = abs(modified_min) % delta

        if mismatch < (delta / 2.):
            nudge = mismatch
        else:
            nudge = mismatch - delta

        modified_min += nudge
        modified_max += nudge

    return modified_min, modified_max


def get_tensor_min_max(t, per_dim=None):
    if per_dim is None:
        return t.min(), t.max()
    if per_dim > t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.min(dim=-1)[0], tv.max(dim=-1)[0]


def symmetric_quant_func(x, k, x_mag):
    """
    inputs: TensorT, int, float
    outputs: TensorT, float, TensorT

    """
    assert 0 < x_mag

    x_min, x_max = -x_mag, x_mag
    idx = (x_min <= x) * (x <= x_max)
    x = torch.clamp(x, x_min, x_max)
    n = 2 ** (k - 1) - 1
    q_d = x_max / n
    q_i = torch.round(x / q_d)

    return q_i, q_d, idx


class AsymmetricQuantFunction(Function):
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None, per_channel=False, percentile_mode=False, show=False):
        if x_min is None or x_max is None:
            x_min, x_max = x.min(), x.max()

        if per_channel:
            scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)

            new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)

            # Need to clamp x if percentile mode is True
            n = 2 ** k - 1
            new_quant_x = torch.clamp(new_quant_x, 0, n)

            quant_x = linear_dequantize(new_quant_x, scale, zero_point, inplace=False)

            if show:
                return torch.autograd.Variable(quant_x), scale, zero_point
            else: 
                return torch.autograd.Variable(quant_x)
        else:
            scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max)
            new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
            quant_x = linear_dequantize(new_quant_x, scale, zero_point, inplace=False)
            if show:
                return torch.autograd.Variable(quant_x), scale, zero_point
            else:
                return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None, None, None, None


class SymmetricQuantFunction(Function):
    @staticmethod
    def forward(ctx, x, k, x_min=None, x_max=None, per_channel=False, percentile_mode=False, show=False):
        if per_channel:
            magnitude = torch.max(torch.stack([x_min.abs(), x_max.abs()], dim=1), dim=1).values
        else:
            magnitude = max(x_min.abs(), x_max.abs())
        scale, zero_point = symmetric_linear_quantization_params(k, magnitude)
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)

        n = 2 ** (k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n-1)

        quant_x = linear_dequantize(new_quant_x, scale, zero_point, inplace=False)
        
        if show:
            return torch.autograd.Variable(quant_x), scale, zero_point
        else:
            return torch.autograd.Variable(quant_x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None, None, None, None, None
