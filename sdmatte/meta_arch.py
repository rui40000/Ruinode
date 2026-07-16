# -*- coding: utf-8 -*-
"""
SDMatte 主干网络。

移植自 vivoCameraResearch/SDMatte 的 modeling/SDMatte/meta_arch.py。
相对官方版本只做了两类改动，计算图本身逐行保持一致：

1. 官方把 .cuda() 硬编码在 forward 各处，这里改为跟随模型自身所在设备，
   以便在 ComfyUI 里支持 CPU / 多卡 / 显存卸载。
2. 官方 init_submodule 在 load_weight=False 时用 os.path.join 拼子目录，
   这里同样只读配置、不读权重，但对路径做了存在性校验并给出可读的报错。

注意 load_weight 恒为 False 是官方推理配置（configs/SDMatte.py）的原意：
网络结构只从 config.json 构建，全部权重随后由 checkpoint 覆盖，
因此 Stable Diffusion 2.1 的权重文件在整个流程中不参与。
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.models.embeddings import get_timestep_embedding
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextConfig

from .nn_utils import replace_unet_conv_in, replace_attention_mask_method, add_aux_conv_in
from .replace import CustomUNet

AUX_INPUT_DIT = {
    "auto_mask": "auto_coords",
    "point_mask": "point_coords",
    "bbox_mask": "bbox_coords",
    "mask": "mask_coords",
    "trimap": "trimap_coords",
}


class SDMatte(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        conv_scale=3,
        num_inference_steps=1,
        aux_input="bbox_mask",
        use_aux_input=False,
        use_coor_input=True,
        use_dis_loss=True,
        use_attention_mask=True,
        use_encoder_attention_mask=False,
        add_noise=False,
        attn_mask_aux_input=["point_mask", "bbox_mask", "mask"],
        aux_input_list=["point_mask", "bbox_mask", "mask"],
        use_encoder_hidden_states=True,
        residual_connection=False,
        use_attention_mask_list=[True, True, True],
        use_encoder_hidden_states_list=[True, True, True],
        load_weight=True,
    ):
        super().__init__()
        self.init_submodule(pretrained_model_name_or_path, load_weight)
        self.num_inference_steps = num_inference_steps
        self.aux_input = aux_input
        self.use_aux_input = use_aux_input
        self.use_coor_input = use_coor_input
        self.use_dis_loss = use_dis_loss
        self.use_attention_mask = use_attention_mask
        self.use_encoder_attention_mask = use_encoder_attention_mask
        self.add_noise = add_noise
        self.attn_mask_aux_input = attn_mask_aux_input
        self.aux_input_list = aux_input_list
        self.use_encoder_hidden_states = use_encoder_hidden_states
        if use_encoder_hidden_states:
            self.unet = add_aux_conv_in(self.unet)
        if not add_noise:
            conv_scale -= 1
        if not use_aux_input:
            conv_scale -= 1
        if conv_scale > 1:
            self.unet = replace_unet_conv_in(self.unet, conv_scale)
        replace_attention_mask_method(self.unet, residual_connection)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.use_attention_mask_list = use_attention_mask_list
        self.unet.use_encoder_hidden_states_list = use_encoder_hidden_states_list

    @property
    def device(self):
        return next(self.unet.parameters()).device

    def init_submodule(self, pretrained_model_name_or_path, load_weight):
        if load_weight:
            self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
            self.unet = CustomUNet.from_pretrained(
                pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=True, ignore_mismatched_sizes=False
            )
            self.noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
            return

        # 仅从 config 构建骨架，不加载任何 SD 权重
        unet_path = os.path.join(pretrained_model_name_or_path, "unet")
        unet_cfg_file = os.path.join(unet_path, "config.json")
        if not os.path.isfile(unet_cfg_file):
            raise FileNotFoundError(f"缺少 SDMatte 的 unet 配置：{unet_cfg_file}")

        unet_config = CustomUNet.load_config(unet_path)
        # SDMatte 在标准 SD 2.1 的 UNet 配置上追加了三个字段，用于坐标嵌入与不透明度嵌入。
        # 缺任何一个都说明配置来源不对（例如误用了原版 SD 2.1 的 config.json），
        # 此时若放任默认值继续，权重会因形状不符被静默丢弃，必须直接报错。
        for field in ("bbox_time_embed_dim", "point_embeddings_input_dim", "bbox_embeddings_input_dim"):
            if field not in unet_config:
                raise ValueError(
                    f"unet 配置缺少 SDMatte 专有字段 '{field}'：{unet_cfg_file}\n"
                    "这通常是误用了原版 Stable Diffusion 2.1 的 config.json。"
                    "请使用 LongfeiHuang/SDMatte 提供的配置文件。"
                )

        text_config = CLIPTextConfig.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder = CLIPTextModel(text_config)

        vae_path = os.path.join(pretrained_model_name_or_path, "vae")
        self.vae = AutoencoderKL.from_config(AutoencoderKL.load_config(vae_path))

        self.unet = CustomUNet.from_config(unet_config, low_cpu_mem_usage=True, ignore_mismatched_sizes=False)

        scheduler_path = os.path.join(pretrained_model_name_or_path, "scheduler", "scheduler_config.json")
        self.noise_scheduler = DDIMScheduler.from_config(DDIMScheduler.load_config(scheduler_path))

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

    def forward(self, data):
        device = self.device
        rgb = data["image"].to(device)
        B = rgb.shape[0]

        if self.aux_input is None and self.training:
            aux_input_type = random.choice(self.aux_input_list)
        elif self.aux_input is None:
            aux_input_type = "point_mask"
        else:
            aux_input_type = self.aux_input

        # 视觉提示 -> latent
        if self.use_aux_input:
            aux_input = data[aux_input_type].to(device)
            aux_input = aux_input.repeat(1, 3, 1, 1)
            aux_input_h = self.vae.encoder(aux_input.to(rgb.dtype))
            aux_input_moments = self.vae.quant_conv(aux_input_h)
            aux_input_mean, _ = torch.chunk(aux_input_moments, 2, dim=1)
            aux_input_latent = aux_input_mean * self.vae.config.scaling_factor
        else:
            aux_input_latent = None

        # 视觉提示的坐标嵌入
        coor_name = AUX_INPUT_DIT[aux_input_type]
        coor = data[coor_name].to(device)
        if coor_name == "point_coords":
            N = coor.shape[1]
            for i in range(N, 1680):
                if 1680 % i == 0:
                    num_channels = 1680 // i
                    pad_size = i - N
                    padding = torch.zeros((B, pad_size), dtype=coor.dtype, device=coor.device)
                    coor = torch.cat([coor, padding], dim=1)
                    zero_coor = torch.zeros((B, pad_size + N), dtype=coor.dtype, device=coor.device)
                    break
            if self.use_coor_input:
                coor = get_timestep_embedding(
                    coor.flatten(),
                    num_channels,
                    flip_sin_to_cos=True,
                    downscale_freq_shift=0,
                )
            else:
                coor = get_timestep_embedding(
                    zero_coor.flatten(),
                    num_channels,
                    flip_sin_to_cos=True,
                    downscale_freq_shift=0,
                )
            added_cond_kwargs = {"point_coords": coor}
        else:
            if self.use_coor_input:
                added_cond_kwargs = {"bbox_mask_coords": coor}
            else:
                coor = torch.tensor([[0, 0, 1, 1]] * B, device=device)
                added_cond_kwargs = {"bbox_mask_coords": coor}

        # 掩码自注意力
        if self.use_attention_mask and aux_input_type in self.attn_mask_aux_input:
            attention_mask = data[aux_input_type].to(device)
            attention_mask = (attention_mask + 1) / 2
            attention_mask = F.interpolate(attention_mask, scale_factor=1 / 8, mode="nearest")
            attention_mask = attention_mask.flatten(start_dim=1)
        else:
            attention_mask = None

        # 原图 -> latent
        rgb_h = self.vae.encoder(rgb)
        rgb_moments = self.vae.quant_conv(rgb_h)
        rgb_mean, _ = torch.chunk(rgb_moments, 2, dim=1)
        rgb_latent = rgb_mean * self.vae.config.scaling_factor

        # 视觉提示驱动的交互条件
        encoder_hidden_states = None
        if self.use_encoder_hidden_states and aux_input_latent is not None:
            encoder_hidden_states = self.unet.aux_conv_in(aux_input_latent)
            encoder_hidden_states = encoder_hidden_states.view(B, 1024, -1)
            encoder_hidden_states = encoder_hidden_states.permute(0, 2, 1)

        if "caption" in data:
            prompt = data["caption"]
        else:
            prompt = [""] * B
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        text_embed = self.text_encoder(text_input_ids)[0]
        encoder_hidden_states_2 = text_embed

        # 不透明度嵌入：透明物体走另一条分支
        is_trans = data["is_trans"].to(device)
        trans = 1 - is_trans

        # 官方在此处构造了 timestep 张量却传入 None，即单步、不加噪。此处保持一致。
        label_latent = self.unet(
            sample=torch.cat([rgb_latent, aux_input_latent], dim=1),
            trans=trans,
            timestep=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_2=encoder_hidden_states_2,
            added_cond_kwargs=added_cond_kwargs,
            attention_mask=attention_mask,
        ).sample
        label_latent = label_latent / self.vae.config.scaling_factor
        z = self.vae.post_quant_conv(label_latent)
        stacked = self.vae.decoder(z)
        label_mean = stacked.mean(dim=1, keepdim=True)
        output = torch.clip(label_mean, -1.0, 1.0)
        output = (output + 1.0) / 2.0
        return output
