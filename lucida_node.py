# -*- coding: utf-8 -*-
"""
Lucida 抠图节点（Ruinode）
==========================
模型：https://huggingface.co/egeorcun/lucida （MIT）
项目：https://github.com/egeorcun/lucida

Lucida 是 BiRefNet_HR 的微调版，训练目标是攻克多数开源抠图模型的短板：
伪装物体、透明材质（玻璃）、文字与 Logo、VFX 光效、插画。
作者在 203 图 9 类别基准上的 MAE（越低越好）：
伪装 0.0270、插画 0.0092、文字/Logo 0.0091（商业参考为 0.0123）、
印刷设计 0.0235、总体 0.0257。

与本仓库另两个抠图节点的分工：
- Lucida ：全自动。文字/Logo、插画、玻璃、伪装物体场景下最强
- FeyNobg：全自动。常规主体照片，速度快
- SDMatte：需框/掩码提示，适合画面中多个主体、只抠其中一个

实现说明：
- 模型代码内嵌在 lucida/ 子包（见其 __init__.py），不使用 trust_remote_code，
  避免运行时联网执行远程代码；
- 预处理与 FeyNobg 完全一致（1024 双线性 + ImageNet 标准化），
  直接复用那份已验证过的实现，不重复造轮子；
- 模型内部 Config.size=1024 且 dec_ipt_split=True，decoder 与 1024 输入绑定，
  因此不开放分辨率选项（实测非 1024 会因通道数不匹配报错）。
"""
import os

import numpy as np
import torch

import folder_paths

# ---------------------------------------------------------------- 模型目录注册

LUCIDA_DIR = os.path.join(folder_paths.models_dir, "lucida")
os.makedirs(LUCIDA_DIR, exist_ok=True)

if "lucida" in folder_paths.folder_names_and_paths:
    _paths, _exts = folder_paths.folder_names_and_paths["lucida"]
    if LUCIDA_DIR not in _paths:
        _paths.append(LUCIDA_DIR)
    _exts.update({".safetensors", ".pth", ".pt"})
else:
    folder_paths.folder_names_and_paths["lucida"] = (
        [LUCIDA_DIR], {".safetensors", ".pth", ".pt"})

HF_REPO = "egeorcun/lucida"
DEFAULT_CKPT = "lucida.safetensors"
_NO_MODEL_HINT = "未找到权重，将自动下载"

_MODEL_CACHE = {}


def _list_ckpts():
    files = []
    if os.path.isdir(LUCIDA_DIR):
        files = sorted(f for f in os.listdir(LUCIDA_DIR)
                       if f.lower().endswith((".safetensors", ".pth", ".pt")))
    return files or [_NO_MODEL_HINT]


def _ensure_ckpt(name):
    """返回权重路径；没有就从 HuggingFace 拉一份（约 885MB）。"""
    if name and name != _NO_MODEL_HINT:
        p = os.path.join(LUCIDA_DIR, name)
        if os.path.isfile(p):
            return p

    target = os.path.join(LUCIDA_DIR, DEFAULT_CKPT)
    if os.path.isfile(target):
        return target

    print(f"[Ruinode-Lucida] 本地未找到权重，开始从 HuggingFace 下载 {HF_REPO}"
          f"（约 885MB）→ {target}")
    try:
        from huggingface_hub import hf_hub_download

        got = hf_hub_download(HF_REPO, "model.safetensors", local_dir=LUCIDA_DIR)
        src = os.path.join(LUCIDA_DIR, "model.safetensors")
        if os.path.isfile(src) and not os.path.isfile(target):
            os.rename(src, target)
        elif os.path.isfile(got) and not os.path.isfile(target):
            target = got
    except Exception as e:
        raise RuntimeError(
            f"自动下载失败：{e}\n"
            f"请手动下载 https://huggingface.co/{HF_REPO} 的 model.safetensors，"
            f"重命名为 {DEFAULT_CKPT} 放到：{LUCIDA_DIR}"
        )
    if not os.path.isfile(target):
        raise RuntimeError(f"下载完成但未找到权重文件：{target}")
    print(f"[Ruinode-Lucida] 下载完成：{target}")
    return target


def _load_weights_strict(model, ckpt_path):
    """加载权重并严格校验，绝不接受"能跑但结果劣化"的静默失配。"""
    if ckpt_path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd

    # transformers 保存时可能带 model. 前缀，去掉后再比对
    own = set(model.state_dict().keys())
    if len(own & set(sd.keys())) == 0:
        stripped = {k.split("model.", 1)[-1] if k.startswith("model.") else k: v
                    for k, v in sd.items()}
        if len(own & set(stripped.keys())) > 0:
            print("[Ruinode-Lucida] 权重键名去掉 'model.' 前缀后对齐")
            sd = stripped

    result = model.load_state_dict(sd, strict=False)
    # BiRefNet 的 Swin 会注册 relative_position_index / attn_mask 一类
    # 由窗口尺寸推出的确定性 buffer，构造时已算好，不需要也不该来自权重
    benign = ("relative_position_index", "attn_mask")
    bad_missing = [k for k in result.missing_keys if not k.endswith(benign)]
    bad_unexpected = list(result.unexpected_keys)
    if bad_missing or bad_unexpected:
        for k in bad_missing[:8]:
            print(f"    未被覆盖: {k}")
        for k in bad_unexpected[:8]:
            print(f"    未被使用: {k}")
        raise RuntimeError(
            f"权重与网络结构不匹配（异常缺失 {len(bad_missing)}，异常多余 "
            f"{len(bad_unexpected)}）。继续推理只会得到劣化结果，故中止。\n"
            f"请确认权重确为 egeorcun/lucida 的 model.safetensors。"
        )
    print(f"[Ruinode-Lucida] 权重加载完成（{len(own) - len(result.missing_keys)}"
          f"/{len(own)} 个张量，另 {len(result.missing_keys)} 个为确定性 buffer）")


def _load_model(ckpt_path, dtype, device):
    key = (ckpt_path, str(dtype), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    from .lucida import BiRefNet, BiRefNetConfig

    print(f"[Ruinode-Lucida] 加载模型：{os.path.basename(ckpt_path)}"
          f"（{dtype}，{device}）")
    _MODEL_CACHE.clear()  # 单份 880MB 起，不做多份缓存
    # bb_pretrained=False：只搭骨架，不去联网拉 Swin 的 ImageNet 预训练权重
    model = BiRefNet(config=BiRefNetConfig(bb_pretrained=False))
    _load_weights_strict(model, ckpt_path)
    model.eval()
    model.to(device=device, dtype=dtype)
    _MODEL_CACHE[key] = model
    return model


class RuiLucida:
    """Lucida 全自动抠图：擅长文字/Logo、插画、玻璃与伪装物体。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (_list_ckpts(), {
                    "tooltip": "放在 models/lucida 下的权重（.safetensors）。\n"
                               "未找到时首次运行会自动从 HuggingFace 下载"
                               "egeorcun/lucida（约 885MB）。"
                }),
                "precision": (["fp32", "fp16"], {
                    "default": "fp16",
                    "tooltip": "fp16 显存减半、速度更快，实测与 fp32 输出基本一致。\n"
                               "遇到黑图或异常时改回 fp32。"
                }),
                "device": (["auto", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "alpha_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "多大置信度才算前景。模型对拿不准的区域会输出 0.5 上下的\n"
                               "中间值，表现为「整片主体半透明发灰」，调低可拉回不透明。\n"
                               "只对已有一定响应的区域有效。"
                }),
                "alpha_softness": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "阈值两侧过渡带宽度，决定边缘软硬。\n"
                               "1.0 = 完全不处理，原样输出模型结果（默认）\n"
                               "0.2~0.4 = 压掉灰雾但保留发丝级过渡\n"
                               "0 = 硬二值化。Lucida 强项之一是玻璃等真实半透明，\n"
                               "调小会把这类效果一并压掉，务必按素材取舍。"
                }),
                "keep_aspect_ratio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "模型固定吃 1024×1024，默认把图直接拉伸成正方形\n"
                               "（与官方用法一致）。长图/宽图形变严重时可开启，\n"
                               "改为等比缩放 + 边缘延展补边，推理后裁掉补边。\n"
                               "属试验性选项，常规比例建议关闭。"
                }),
                "invert_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "反转 alpha：默认前景为白（1），开启后前景为黑。"
                }),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("alpha", "cutout")
    FUNCTION = "matting"
    CATEGORY = "Rui-Node🐶/抠图✂️"

    @classmethod
    def VALIDATE_INPUTS(cls, model_name, **kwargs):
        """权重列表随目录变化，宽松放行，运行时兜底（含自动下载）。"""
        return True

    @staticmethod
    def _remap_alpha(alpha, threshold, softness):
        """按阈值/柔和度给遮罩拉一次色阶；默认 (0.5, 1.0) 时原样返回。"""
        low = threshold - softness / 2.0
        high = threshold + softness / 2.0
        if high <= low:
            return (alpha >= threshold).to(alpha.dtype)
        if abs(low) < 1e-6 and abs(high - 1.0) < 1e-6:
            return alpha
        return ((alpha - low) / (high - low)).clamp(0.0, 1.0)

    def matting(self, image, model_name, precision, device,
                alpha_threshold=0.5, alpha_softness=1.0,
                keep_aspect_ratio=False, invert_mask=False):
        import comfy.model_management

        # BiRefNet 系模型的预处理规格相同（1024 + ImageNet 标准化），
        # 直接复用 FeyNobg 节点里那份已验证的实现
        from .feynobg import BiRefNetImageProcessor

        ckpt = _ensure_ckpt(model_name)
        dev = torch.device("cpu") if device == "cpu" \
            else comfy.model_management.get_torch_device()
        dtype = torch.float32 if precision == "fp32" else torch.float16
        model = _load_model(ckpt, dtype, dev)

        proc = BiRefNetImageProcessor(size={"height": 1024, "width": 1024})

        B, H, W, C = image.shape
        if C == 4:
            image = image[..., :3]
        elif C == 1:
            image = image.repeat(1, 1, 1, 3)
        elif C != 3:
            raise ValueError(f"image 需要 1 / 3 / 4 通道，实际收到 {C} 通道")
        image = image.contiguous()

        alphas = []
        for i in range(B):
            # 逐张推理：Swin-Large 在 1024 下峰值显存不低，整批一次容易 OOM
            pixel_values, meta = proc.preprocess_tensor(
                image[i:i + 1], device=dev, dtype=dtype,
                keep_aspect=bool(keep_aspect_ratio))
            with torch.no_grad():
                preds = model(pixel_values)
            # eval 模式下 BiRefNet.forward 返回各尺度预测的列表，最后一项是最终结果
            logits = preds[-1] if isinstance(preds, (list, tuple)) else preds
            alpha = proc.post_process_alpha_matting(
                {"logits": logits.float()}, target_sizes=[(H, W)], crop=meta)[0]
            alphas.append(alpha.clamp(0, 1).cpu())

        alpha = torch.stack(alphas)  # [B,H,W]
        alpha = self._remap_alpha(alpha, float(alpha_threshold),
                                  float(alpha_softness))
        if invert_mask:
            alpha = 1.0 - alpha

        cutout = image.detach().cpu().float() * alpha.unsqueeze(-1)
        return (alpha, cutout)


NODE_CLASS_MAPPINGS = {
    "RuiLucida": RuiLucida,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiLucida": "Lucida 抠图 / Lucida Matting",
}
