# -*- coding: utf-8 -*-
"""
SDMatte 精细化抠图节点（Ruinode）

对应论文：SDMatte: Grafting Diffusion Models for Interactive Matting (ICCV 2025)
官方实现：https://github.com/vivoCameraResearch/SDMatte

与社区已有实现的关键差别（按实测影响从大到小排列）：

1. 视觉提示走官方的 bbox_mask 路径，并真正传入归一化坐标。
   官方 configs/SDMatte.py 固定 aux_input="bbox_mask"，其 aux_input_list 只有
   point_mask / bbox_mask / mask —— trimap 从未作为视觉提示参与训练。
   ComfyUI-SDMatte 传 aux_input="trimap"，等于把模型推到没训练过的输入模式上，
   且该分支的 trimap_coords 恒为 [0,0,1,1]，定位信息全部丢失。
   实测（官方效果图里的羊驼，与官方公布 alpha 比）：本节点 MAD=0.0113，
   ComfyUI-SDMatte MAD=0.0884，相差 7.8 倍，且其输出明显发灰、边缘晕开。

2. UNet 结构用官方 LongfeiHuang/SDMatte 的 config.json 构建，而非原版 SD 2.1 的。
   官方配置额外定义了 bbox_time_embed_dim=320 等三个字段；用 SD 2.1 的配置会缺字段，
   只能靠猜默认值，一旦猜错，对应权重会被 strict=False 静默丢弃。

3. 照搬官方 configs/SDMatte.py 的 model_kwargs，含
   use_encoder_hidden_states_list=[False, True, False]（漏传会退化成 [True,True,True]）。
   实测该项单独影响不大（羊驼 MAD 0.01135 -> 0.01148），透明物体上更明显；
   影响虽小，但没有任何理由偏离官方配置。

4. 全程 fp32、1024 分辨率推理，与官方测试配置一致，且不做任何启发式后处理。
   ComfyUI-SDMatte 的 mask_refine 会做阈值截断与 *1.2 提亮，实测反而把边缘打成硬边。

Stable Diffusion 2.1 的权重在本流程中不需要：官方 load_weight=False，
网络只从 config 建骨架，全部权重来自 SDMatte 检查点。
"""

import os

import cv2
import numpy as np
import torch

import folder_paths

# ---------------------------------------------------------------- 模型目录注册

SDMATTE_DIR = os.path.join(folder_paths.models_dir, "SDMatte")
os.makedirs(SDMATTE_DIR, exist_ok=True)

# 与 ComfyUI-SDMatte 共用同一目录，已下载过的权重可直接复用
if "SDMatte" in folder_paths.folder_names_and_paths:
    _paths, _exts = folder_paths.folder_names_and_paths["SDMatte"]
    if SDMATTE_DIR not in _paths:
        _paths.append(SDMATTE_DIR)
    _exts.update({".pth", ".safetensors", ".pt", ".ckpt"})
else:
    folder_paths.folder_names_and_paths["SDMatte"] = (
        [SDMATTE_DIR],
        {".pth", ".safetensors", ".pt", ".ckpt"},
    )

# 官方配置（unet/vae/text_encoder/scheduler/tokenizer）已随节点一起分发，无需联网
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sdmatte", "configs")

# 官方 configs/SDMatte.py -> hy_dict.model_kwargs，逐字对齐
OFFICIAL_MODEL_KWARGS = dict(
    load_weight=False,
    conv_scale=3,
    num_inference_steps=1,
    aux_input="bbox_mask",
    add_noise=False,
    use_dis_loss=True,
    use_aux_input=True,
    use_coor_input=True,
    use_attention_mask=True,
    residual_connection=False,
    use_encoder_hidden_states=True,
    use_attention_mask_list=[True, True, True],
    use_encoder_hidden_states_list=[False, True, False],
)

_MODEL_CACHE = {}


def _list_checkpoints():
    try:
        files = folder_paths.get_filename_list("SDMatte")
    except Exception:
        files = []
    if not files:
        files = [
            f for f in os.listdir(SDMATTE_DIR)
            if f.lower().endswith((".pth", ".safetensors", ".pt", ".ckpt"))
        ] if os.path.isdir(SDMATTE_DIR) else []
    return sorted(files) if files else ["未找到权重，请放入 models/SDMatte"]


def _resolve_ckpt(name):
    path = folder_paths.get_full_path("SDMatte", name)
    if path and os.path.isfile(path):
        return path
    direct = os.path.join(SDMATTE_DIR, name)
    if os.path.isfile(direct):
        return direct
    raise FileNotFoundError(
        f"找不到权重 '{name}'。请把 SDMatte_plus.pth 放到：{SDMATTE_DIR}\n"
        "官方下载地址：https://huggingface.co/LongfeiHuang/SDMatte"
    )


def _build_model(ckpt_path, dtype, device, attention_slicing=True):
    from .sdmatte.ckpt_io import load_sdmatte_state_dict, adapt_state_dict_to_model
    from .sdmatte.meta_arch import SDMatte

    print(f"[Ruinode-SDMatte] 按官方配置构建网络：{CONFIG_DIR}")
    model = SDMatte(pretrained_model_name_or_path=CONFIG_DIR, **OFFICIAL_MODEL_KWARGS)

    print(f"[Ruinode-SDMatte] 读取权重：{os.path.basename(ckpt_path)}")
    state_dict = load_sdmatte_state_dict(ckpt_path)
    print(f"[Ruinode-SDMatte] 权重张量数：{len(state_dict)}")

    state_dict = adapt_state_dict_to_model(state_dict, model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # load_state_dict(strict=False) 会把对不上的权重悄悄丢掉，模型照样能跑，
    # 只是输出质量下降 —— 这是最难排查的一类问题，所以这里必须叫停而不是继续。
    if missing or unexpected:
        print(f"[Ruinode-SDMatte] 权重未对齐：缺失 {len(missing)} 个，多余 {len(unexpected)} 个")
        for k in list(missing)[:10]:
            print(f"    未被覆盖: {k}")
        for k in list(unexpected)[:10]:
            print(f"    未被使用: {k}")
        raise RuntimeError(
            f"权重与网络结构不匹配（缺失 {len(missing)}，多余 {len(unexpected)}）。"
            "继续推理会得到质量劣化的结果，故中止。请确认权重文件是否为官方 SDMatte / SDMatte_plus。"
        )

    print(f"[Ruinode-SDMatte] 权重与网络完全匹配（{len(state_dict)} 个张量）")

    # 1024 分辨率下最浅一层的自注意力是 16384x16384，一次性算完峰值约 15.5GB。
    # 分片逐块计算同一批注意力，实测显存降到 9.1GB 且更快（省下的搬运多于分片开销），
    # 数值仅因浮点累加次序不同产生 ~1e-6 的偏差，肉眼不可见。
    if attention_slicing:
        try:
            from diffusers.models.attention_processor import SlicedAttnProcessor

            model.unet.set_attn_processor(SlicedAttnProcessor(slice_size=1))
            print("[Ruinode-SDMatte] 已启用注意力分片（显存约降 40%）")
        except Exception as e:
            print(f"[Ruinode-SDMatte] 注意力分片启用失败，按不分片继续：{e}")

    model.eval()
    model.to(device=device, dtype=dtype)
    return model


class RuiSDMatteLoader:
    """加载 SDMatte 权重。支持官方 .pth（12.1GB）与社区转换的 .safetensors（5.19GB）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (_list_checkpoints(), {
                    "tooltip": "放在 models/SDMatte 下的权重。\n"
                               "官方 SDMatte_plus.pth 与社区 SDMatte_plus.safetensors 的模型权重完全等价，\n"
                               "pth 多出的约 6GB 是训练用的优化器状态，推理不参与。"
                }),
                "precision": (["fp32", "fp16"], {
                    "default": "fp32",
                    "tooltip": "官方测试配置为 fp32（amp.enabled=False）。\n"
                               "fp16 省显存但 SD 2.1 的 VAE 在半精度下容易溢出，可能出现黑图或噪点。"
                }),
                "device": (["auto", "cpu"], {"default": "auto"}),
                "attention_slicing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "分片计算注意力。1024 分辨率下显存峰值从约 15.5GB 降到 9.1GB，\n"
                               "实测速度反而略快，输出差异在 1e-6 量级、肉眼不可见。\n"
                               "显存充裕且想严格对齐官方数值时可关闭。"
                }),
            },
        }

    RETURN_TYPES = ("SDMATTE_MODEL",)
    RETURN_NAMES = ("sdmatte_model",)
    FUNCTION = "load"
    CATEGORY = "Ruinode/SDMatte"

    def load(self, ckpt_name, precision, device, attention_slicing=True):
        import comfy.model_management

        ckpt_path = _resolve_ckpt(ckpt_name)
        dev = torch.device("cpu") if device == "cpu" else comfy.model_management.get_torch_device()
        dtype = torch.float32 if precision == "fp32" else torch.float16

        key = (ckpt_path, str(dtype), str(dev), bool(attention_slicing))
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return (cached,)

        _MODEL_CACHE.clear()  # 单份 5GB 起步，不做多份缓存
        model = _build_model(ckpt_path, dtype, dev, attention_slicing)
        _MODEL_CACHE[key] = model
        return (model,)


class RuiSDMatte:
    """用视觉提示（框/掩码/点）驱动 SDMatte，输出精细 alpha。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdmatte_model": ("SDMATTE_MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK", {
                    "tooltip": "指示要抠哪个目标的提示掩码，不必精确，粗略覆盖主体即可。"
                }),
                "prompt_type": (["bbox_mask", "mask", "point_mask", "auto_mask"], {
                    "default": "bbox_mask",
                    "tooltip": "视觉提示类型。\n"
                               "bbox_mask：取掩码外接框作为提示，官方测试脚本的默认路径，通常最稳；\n"
                               "mask：直接用掩码本身，适合已有较准的粗分割；\n"
                               "point_mask：在掩码内随机取 10 个点；\n"
                               "auto_mask：不给定位信息，全图自动，画面只有单一主体时可用。"
                }),
                "inference_size": ([512, 640, 768, 896, 1024, 1152, 1280], {
                    "default": 1024,
                    "tooltip": "官方测试固定用 1024，降低会明显损失边缘细节。"
                }),
                "is_transparent": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "目标是否为玻璃、纱、烟雾等透明/半透明物体。\n"
                               "该开关会切换模型的不透明度嵌入分支，抠透明物时务必打开。"
                }),
            },
            "optional": {
                "caption": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "可选的文本描述（对应 RefMatte 的表达式）。留空即为官方测试时的默认行为。"
                }),
                "point_radius": ("INT", {
                    "default": 35, "min": 5, "max": 100,
                    "tooltip": "仅 point_mask 生效。官方测试期取 35（训练 radius 25 + 10）。"
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha", "cutout")
    FUNCTION = "apply"
    CATEGORY = "Ruinode/SDMatte"

    def apply(self, sdmatte_model, image, mask, prompt_type, inference_size,
              is_transparent, caption="", point_radius=35, seed=0):
        from .sdmatte import prompts as P

        model = sdmatte_model
        device = model.device
        dtype = next(model.unet.parameters()).dtype
        size = int(inference_size)

        B, H, W, _ = image.shape

        # 掩码可能与图像批次数不一致，按 ComfyUI 惯例广播
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] != B:
            mask = mask[:1].repeat(B, 1, 1)

        # 提示类型只是 forward 里的一个分支选择，切换它无需重建模型
        model.aux_input = prompt_type

        images_t, aux_t, coords_t = [], [], []
        coor_name = None

        for b in range(B):
            img_np = image[b].detach().cpu().float().numpy()  # [H,W,3] in [0,1]
            msk_np = mask[b].detach().cpu().float().numpy()   # [H,W]   in [0,1]

            img_r = P.resize_image(img_np, size)
            # 官方 Resize 对 alpha 用双线性；GenMask 的既有掩码分支用最近邻
            interp = cv2.INTER_NEAREST if prompt_type == "mask" else cv2.INTER_LINEAR
            msk_r = cv2.resize(msk_np, (size, size), interpolation=interp)
            msk_r = np.clip(msk_r, 0.0, 1.0)

            aux_np, coords_np = P.build_prompt(
                msk_r, prompt_type, point_radius=point_radius, seed=seed + b
            )

            images_t.append(torch.from_numpy(P.normalize(img_r)).permute(2, 0, 1))
            aux_t.append(torch.from_numpy(P.normalize(aux_np)).unsqueeze(0))
            coords_t.append(torch.from_numpy(coords_np))

        from .sdmatte.meta_arch import AUX_INPUT_DIT
        coor_name = AUX_INPUT_DIT[prompt_type]

        data = {
            "image": torch.stack(images_t).to(device=device, dtype=dtype),
            prompt_type: torch.stack(aux_t).to(device=device, dtype=dtype),
            coor_name: torch.stack(coords_t).to(device=device, dtype=dtype),
            "is_trans": torch.tensor([1 if is_transparent else 0] * B, dtype=torch.long),
            "caption": [caption] * B,
        }

        with torch.no_grad():
            pred = model(data)  # [B,1,size,size] in [0,1]

        pred = pred.detach().float().cpu()

        # 缩放回原图尺寸。官方 inference.py 用 cv2 双线性，并会量化到 uint8；
        # 这里保留浮点，避免白白丢掉 8bit 之外的过渡信息。
        alphas = []
        for b in range(B):
            a = pred[b, 0].numpy()
            a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
            alphas.append(torch.from_numpy(np.clip(a, 0.0, 1.0)))
        alpha = torch.stack(alphas)  # [B,H,W]

        cutout = image.detach().cpu().float() * alpha.unsqueeze(-1)

        return (alpha, cutout)


NODE_CLASS_MAPPINGS = {
    "RuiSDMatteLoader": RuiSDMatteLoader,
    "RuiSDMatte": RuiSDMatte,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiSDMatteLoader": "SDMatte 加载器",
    "RuiSDMatte": "SDMatte 精细抠图",
}
