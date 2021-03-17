import torch
import torch.nn as nn
from ..quant_modules import QuantAct, Quant_Conv2d, QuantBnConv2d, QuantBaseNode, \
    QuantDepthwiseNode, QuantDeformConvWithOffsetScaleBoundPositive, QuantBaseNodeDeform


def quantize_shufflenetv2_dcn(model, quant_conv, quant_bn, quant_act, wt_quant_mode, act_quant_mode, wt_per_channel,
                              wt_percentile, act_percentile, deform_backbone, w2=False, maxpool=False):
    """
    quantize DCN with ShuffleNetv2 (PyTorchCV version) as backbone.

    model: the model to be quantized
    quant_conv: bitwidth for weight quantization
    quant_bn: bitwidth for bn quantization
    quant_act: bitwidth for activation quantization
    wt_quant_mode: symmetric or asymmetric mode for weight quantization
    act_quant_mode: symmetric or asymmetric mode for weight quantization
    wt_per_channel: whether to use channel-wise for weight quantization
    wt_percentile: whether to use percentile mode to determine weight quantization range
    act_percentile: whether to use percentile mode to determine activation quantization range
    deform_backbone: whether to use deformable convolution inside the backbone
    w2: if true, increase the channel number of the backbone by 2 times
    maxpool: if true, use stride 2 + maxpool for layer0, otherwise use stride 4
    
    """
    layer0 = getattr(model, 'layer0')
    conv, bn, activ = layer0[0], layer0[1], layer0[2]
    quant_layer0 = QuantBnConv2d(8, quant_mode=wt_quant_mode, per_channel=wt_per_channel, weight_percentile=wt_percentile)
    quant_layer0.set_param(conv, bn)

    if maxpool:
        quant_act0 = nn.Sequential(*[activ, QuantAct(quant_act, quant_mode="asymmetric", percentile=act_percentile), layer0[3]])
    else:
        quant_act0 = nn.Sequential(*[activ, QuantAct(quant_act, quant_mode="asymmetric", percentile=act_percentile)])
    setattr(model, 'layer0', nn.Sequential(*[quant_layer0, quant_act0]))

    for layer_num in range(1, 4):
        mods = []
        layer = getattr(model, 'layer' + str(layer_num))
        share_act = QuantAct(quant_act, quant_mode="asymmetric", percentile=act_percentile)
        for node in layer.children():
            if not deform_backbone:
                quant_node = QuantBaseNode(quant_conv, quant_act, act_percentile=act_percentile,
                                           wt_quant_mode=wt_quant_mode, act_quant_mode=act_quant_mode,
                                           per_channel=wt_per_channel, weight_percentile=wt_percentile)
            else:
                quant_node = QuantBaseNodeDeform(quant_conv, quant_act, act_percentile=act_percentile,
                                                 wt_quant_mode=wt_quant_mode, act_quant_mode=act_quant_mode,
                                                 per_channel=wt_per_channel, weight_percentile=wt_percentile)
            quant_node.set_param(node)
            quant_node.set_act(share_act)
            mods.append(quant_node)
        setattr(model, 'layer' + str(layer_num), nn.Sequential(*mods))

    layer4 = getattr(model, 'layer4')
    conv, bn, activ = layer4[0], layer4[1], layer4[2]
    quant_layer4 = QuantBnConv2d(quant_conv, quant_mode=wt_quant_mode, per_channel=wt_per_channel, weight_percentile=wt_percentile)
    quant_layer4.set_param(conv, bn)
    quant_act4 = nn.Sequential(*[activ, QuantAct(quant_act, quant_mode="asymmetric", percentile=act_percentile)])
    setattr(model, 'layer4', nn.Sequential(*[quant_layer4, quant_act4]))

    for head in model.heads:
        head_mod = getattr(model, head)
        quant_head = QuantDepthwiseNode(quant_conv, quant_act, act_percentile=act_percentile,
                                        wt_quant_mode=wt_quant_mode, act_quant_mode=act_quant_mode,
                                        per_channel=wt_per_channel, weight_percentile=wt_percentile)
        quant_head.set_param(head_mod)
        setattr(model, head, quant_head)

    deform = getattr(model, 'deconv_layers')
    deform_mods = []
    for deform_num in range(0, 3):
        deform_conv, deform_bn = deform[4*deform_num], deform[4*deform_num+1]
        quant_deform = QuantDeformConvWithOffsetScaleBoundPositive(quant_conv, quant_act, act_percentile=act_percentile,
                                                wt_quant_mode=wt_quant_mode, act_quant_mode=act_quant_mode,
                                                per_channel=wt_per_channel, weight_percentile=wt_percentile)
        quant_deform.set_param(deform_conv, deform_bn)
        deform_mods.append(quant_deform)
        quant_deform_act = nn.Sequential(*[deform[4*deform_num+2], QuantAct(quant_act, quant_mode="asymmetric", percentile=act_percentile)])
        deform_mods.append(quant_deform_act)
        deform_mods.append(deform[4*deform_num+3])
    setattr(model, 'deconv_layers', nn.Sequential(*deform_mods))
