import torch
import torch.nn as nn
from ..quant_modules import QuantAct, Quant_Conv2d, QuantBnConv2d, QuantBaseNode, \
    QuantDepthwiseNode, QuantDeformConvWithOffsetScaleBoundPositive, QuantBaseNodeDeform


def quantize_shufflenetv2_dcn(model, quant_conv, quant_bn, quant_act, wt_quant_mode, act_quant_mode, wt_per_channel,
                              wt_percentile, act_percentile, deform_backbone):
    # quantize DCN with ShuffleNetv2 (pytorchcv version) as backbone
    # quantize each components in DCN including backbone and heads.
    layer0 = getattr(model, 'layer0')
    conv, bn, activ = layer0[0], layer0[1], layer0[2]
    quant_layer0 = QuantBnConv2d(8, quant_mode=wt_quant_mode, per_channel=wt_per_channel, weight_percentile=wt_percentile)
    quant_layer0.set_param(conv, bn)
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

