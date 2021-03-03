import torch
import torch.nn as nn
from ..quant_modules import QuantAct, Quant_Conv2d, QuantBnConv2d, QuantSflUnit, QuantBaseNode, QuantHeadNode, \
    QuantDepthwiseNode, QuantDeformConvWithOffsetScaleBoundPositive, QuantBaseNodeDeform
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock


def mix_quantize_model(model, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile, prefix, config):
    if type(model) == nn.Conv2d:
        if config[prefix] is None:
            return model
        quant_mod = Quant_Conv2d(weight_bit=config[prefix], quant_mode=quant_mode, per_channel=wt_per_channel, weight_percentile=wt_percentile)
        quant_mod.set_param(model)
        return quant_mod
    # if type(model) == ConvBlock:
    #     conv, bn = model.conv, model.bn
    #     quant_ConvBn = QuantBnConv2d(weight_bit=quant_conv, quant_mode=quant_mode, full_precision_flag=False)
    #     quant_ConvBn.set_param(conv, bn)
    #     if model.activate:
    #         quant_activ = nn.Sequential(*[model.activ, QuantAct(activation_bit=quant_act, quant_mode=quant_mode, percentile=act_percentile)])
    #         quant_ConvBn = nn.Sequential(*[quant_ConvBn, quant_activ])
    #     return quant_ConvBn
    # elif type(model) == ShuffleUnit:
    #     quant_ShuffleUnit = QuantSflUnit(weight_bit=quant_conv, act_bit=quant_act, quant_mode=quant_mode, per_channel=wt_per_channel)
    #     quant_ShuffleUnit.set_param(model)
    #     return quant_ShuffleUnit
    elif type(model) == nn.Sequential:
        mods = []
        if prefix != '':
            prefix += '.'
        for n, m in model.named_children():
            mods.append(quantize_model(m, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile, prefix + n, config))
        return nn.Sequential(*mods)
    else:
        if prefix != '':
            prefix += '.'
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(model, attr, quantize_model(mod, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile, prefix + attr, config))
        return model


def quantize_model(model, quant_conv, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile):
    if type(model) == nn.Conv2d:
        quant_mod = Quant_Conv2d(weight_bit=quant_conv, quant_mode=quant_mode, per_channel=wt_per_channel, weight_percentile=wt_percentile)
        quant_mod.set_param(model)
        return quant_mod
    # if type(model) == ConvBlock:
    #     conv, bn = model.conv, model.bn
    #     quant_ConvBn = QuantBnConv2d(weight_bit=quant_conv, quant_mode=quant_mode, full_precision_flag=False)
    #     quant_ConvBn.set_param(conv, bn)
    #     if model.activate:
    #         quant_activ = nn.Sequential(*[model.activ, QuantAct(activation_bit=quant_act, quant_mode=quant_mode, percentile=act_percentile)])
    #         quant_ConvBn = nn.Sequential(*[quant_ConvBn, quant_activ])
    #     return quant_ConvBn
    # elif type(model) == ShuffleUnit:
    #     quant_ShuffleUnit = QuantSflUnit(weight_bit=quant_conv, act_bit=quant_act, quant_mode=quant_mode, per_channel=wt_per_channel)
    #     quant_ShuffleUnit.set_param(model)
    #     return quant_ShuffleUnit
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m, quant_conv, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile))
        return nn.Sequential(*mods)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(model, attr, quantize_model(mod, quant_conv, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile))
        return model


def list_model_parameters(model):
    conv_names = [n for n, p in model.named_modules() if isinstance(p, nn.Conv2d)]
    relu_names = [n for n, p in model.named_modules() if isinstance(p, nn.ReLU)]
    return conv_names, relu_names


# quantize shuffflenetv2 defined in pytorchcv
def quantize_shufflenetv2(model, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile):
    for attr in dir(model.features):
        # sequentially quantize all the modules in the network
        m = getattr(model.features, attr)
        if not isinstance(m, nn.Module):
            continue
        if type(m) == ShuffleInitBlock:
            # quantize the first block (name=init_block, type=ShuffleInitBlock) of the network
            conv, bn = m.conv.conv, m.conv.bn
            quant_ConvBn = QuantBnConv2d(weight_bit=quant_conv, quant_mode=quant_mode, per_channel=wt_per_channel)
            quant_ConvBn.set_param(conv, bn)
            if m.conv.activate:
                quant_activ = nn.Sequential(*[m.conv.activ, QuantAct(activation_bit=quant_act, quant_mode='asymmetric', percentile=act_percentile)])
                quant_ConvBn = nn.Sequential(*[quant_ConvBn, quant_activ])
            setattr(m, 'conv', quant_ConvBn)
            setattr(model.features, attr, m)
        elif type(m) == ConvBlock:
            # quantize the last block (name=final_block, type=ConvBlock) of the network
            conv, bn = m.conv, m.bn
            quant_ConvBn = QuantBnConv2d(weight_bit=quant_conv, quant_mode=quant_mode, per_channel=wt_per_channel)
            quant_ConvBn.set_param(conv, bn)
            if m.activate:
                quant_activ = nn.Sequential(*[m.activ, QuantAct(activation_bit=quant_act, quant_mode='asymmetric', percentile=act_percentile)])
                quant_ConvBn = nn.Sequential(*[quant_ConvBn, quant_activ])
            setattr(model.features, attr, quant_ConvBn)
        elif type(m) == nn.Sequential:
            # quantize all the rest stages of the network
            mods = []
            # make sure all the sfl_units share the same quant_act function so that the scaling factor and bias could be the same
            share_act = nn.Sequential(*[nn.ReLU(inplace=True), QuantAct(activation_bit=quant_act, quant_mode='asymmetric', percentile=act_percentile)])
            for sfl_unit in m.children():
                # there are several sfl_unit within a single stage
                quant_ShuffleUnit = QuantSflUnit(weight_bit=quant_conv, act_bit=quant_act, act_percentile=act_percentile, quant_mode=quant_mode, per_channel=wt_per_channel)
                quant_ShuffleUnit.set_param(sfl_unit)
                quant_ShuffleUnit.set_act(share_act)
                mods.append(quant_ShuffleUnit)
            setattr(model.features, attr, nn.Sequential(*mods))
    return model


def quantize_shufflenetv2_dcn(model, quant_conv, quant_bn, quant_act, wt_quant_mode, act_quant_mode, wt_per_channel,
                              wt_percentile, act_percentile, deform_backbone):
    # quantize DCN with ShuffleNetv2 (pytorchcv version) as backbone
    # quantize each components in DCN including backbone and heads.
    layer0 = getattr(model, 'layer0')
    conv, bn, activ = layer0[0], layer0[1], layer0[2]
    quant_layer0 = QuantBnConv2d(8, quant_mode=wt_quant_mode, per_channel=wt_per_channel, weight_percentile=wt_percentile)
    # quant_layer0 = QuantBnConv2d(quant_conv, quant_mode=quant_mode, per_channel=wt_per_channel)
    quant_layer0.set_param(conv, bn)
    # quant_act0 = nn.Sequential(*[activ, QuantAct(quant_act, quant_mode="asymmetric", percentile=act_percentile), layer0[3]])
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
        # quant_head = Quant_Conv2d(weight_bit=quant_conv, quant_mode=quant_mode, per_channel=wt_per_channel, weight_percentile=wt_percentile)
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
    # return model


def quantize_sfl_dcn(model, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile):
    # quantize DCN with ShuffleNetv2 (pytorchcv version) as backbone
    # quantize each components in DCN including base (backbone), dla_up, ida_up and heads.
    model.base = quantize_base(model.base, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile)
    model.dla_up = quantize_dla(model.dla_up, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile)
    model.ida_up = quantize_ida(model.ida_up, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile)
    for head in model.heads:
        head_mod = getattr(model, head)
        quant_head = Quant_Conv2d(weight_bit=quant_conv, quant_mode=quant_mode, per_channel=wt_per_channel, weight_percentile=wt_percentile)
        quant_head.set_param(head_mod)
        setattr(model, head, quant_head)
    return model


def quantize_base(model, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile):
    # quantize layer0 which is the init_block, basically the same as line 85-92
    layer0 = getattr(model, 'layer0')
    conv, bn, activ = layer0[0], layer0[1], layer0[2]
    quant_layer0 = QuantBnConv2d(quant_conv, quant_mode=quant_mode, per_channel=wt_per_channel)
    quant_layer0.set_param(conv, bn)
    quant_act0 = nn.Sequential(*[activ, QuantAct(quant_act, quant_mode=quant_mode, percentile=act_percentile)])
    setattr(model, 'layer0', nn.Sequential(*[quant_layer0, quant_act0]))

    # quantize each stage, basically the same as line 102-113
    for layer_num in range(1, 4):
        mods = []
        layer = getattr(model, 'layer' + str(layer_num))
        share_act = QuantAct(quant_act, quant_mode=quant_mode, percentile=act_percentile)
        for node in layer.children():
            quant_node = QuantBaseNode(quant_conv, quant_act, per_channel=wt_per_channel)
            quant_node.set_param(node)
            quant_node.set_act(share_act)
            mods.append(quant_node)
        setattr(model, 'layer' + str(layer_num), nn.Sequential(*mods))

    return model


def quantize_dla(model, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile):
    # quantize dla module
    model.ida_0 = quantize_ida(model.ida_0, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile)
    model.ida_1 = quantize_ida(model.ida_1, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile)
    return model


def quantize_ida(model, quant_conv, quant_bn, quant_act, quant_mode, wt_per_channel, wt_percentile, act_percentile):
    # quantize ida module
    if type(model.proj_0) != nn.Identity:
        quant_proj0 = QuantHeadNode(quant_conv, quant_act, act_percentile=act_percentile, quant_mode=quant_mode, per_channel=wt_per_channel)
        quant_proj0.set_param(model.proj_0)
        quant_proj0.set_act(QuantAct(quant_act, quant_mode=quant_mode, percentile=act_percentile))
        model.proj_0 = quant_proj0

    if type(model.proj_1) != nn.Identity:
        quant_proj1 = QuantHeadNode(quant_conv, quant_act, act_percentile=act_percentile, quant_mode=quant_mode, per_channel=wt_per_channel)
        quant_proj1.set_param(model.proj_1)
        quant_proj1.set_act(QuantAct(quant_act, quant_mode=quant_mode, percentile=act_percentile))
        model.proj_1 = quant_proj1

    quant_node1 = QuantHeadNode(quant_conv, quant_act, act_percentile=act_percentile, quant_mode=quant_mode, per_channel=wt_per_channel)
    quant_node1.set_param(model.node_1)
    quant_node1.set_act(QuantAct(quant_act, quant_mode=quant_mode, percentile=act_percentile))
    model.node_1 = quant_node1

    try:
        quant_node2 = QuantHeadNode(quant_conv, quant_act, act_percentile=act_percentile, quant_mode=quant_mode, per_channel=wt_per_channel)
        quant_node2.set_param(model.node_2)
        quant_node2.set_act(QuantAct(quant_act, quant_mode=quant_mode, percentile=act_percentile))
        model.node_2 = quant_node2
    except AttributeError:
        pass

    return model
