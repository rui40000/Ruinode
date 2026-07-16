# -*- coding: utf-8 -*-
"""
SDMatte 的网络结构改造工具。

摘自 vivoCameraResearch/SDMatte 的 utils/utils.py，仅保留推理必需的三个函数，
去掉了训练期才用到的 get_unknown_tensor_from_pred（其内部硬编码了 .cuda()）。
函数体与官方保持逐行一致，改动会直接影响权重能否对上号。
"""

import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from diffusers.models.attention_processor import Attention, AttnProcessor

from .replace import custom_prepare_attention_mask, custom_get_attention_scores


def replace_unet_conv_in(unet, num):
    """把 conv_in 从 4 通道扩成 4*num 通道，权重按 num 复制并等比缩小。"""
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]
    _weight = _weight.repeat((1, num, 1, 1))
    # half the activation magnitude
    _weight = _weight / num
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(4 * num, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    # 官方此处会改写 unet.config["in_channels"]；新版 diffusers 的 config 是
    # FrozenDict，赋值会抛异常，而该字段在推理期并不被读取，故安全跳过。
    try:
        unet.config["in_channels"] = 4 * num
    except Exception:
        pass
    return unet


def add_aux_conv_in(unet):
    """新增 aux_conv_in：把视觉提示的 latent 编码成 1024 维，充当 cross-attention 的条件。"""
    aux_conv_in = nn.Conv2d(in_channels=4, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    aux_conv_in.weight.data[:320, :, :, :] = unet.conv_in.weight.data.clone()
    aux_conv_in.weight.data[320:, :, :, :] = 0.0
    aux_conv_in.bias.data[:320] = unet.conv_in.bias.data.clone()
    aux_conv_in.bias.data[320:] = 0.0
    unet.aux_conv_in = aux_conv_in
    return unet


def replace_attention_mask_method(module, residual_connection):
    """递归替换注意力的 mask 处理逻辑，使其支持 SDMatte 的空间掩码自注意力。"""
    if isinstance(module, Attention):
        module.processor = AttnProcessor()
        if hasattr(module, "prepare_attention_mask"):
            module.prepare_attention_mask = custom_prepare_attention_mask.__get__(module)
        if hasattr(module, "cross_attention_dim") and module.cross_attention_dim == 320:
            module.residual_connection = residual_connection
        if hasattr(module, "get_attention_scores"):
            module.get_attention_scores = custom_get_attention_scores.__get__(module)

    for child_name, child_module in module.named_children():
        replace_attention_mask_method(child_module, residual_connection)
