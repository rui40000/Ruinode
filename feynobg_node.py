# -*- coding: utf-8 -*-
"""
FeyNobg 抠图节点（Ruinode）
===========================
对应模型：https://huggingface.co/feyninc/FeyNobg （Apache-2.0）
上游库：  https://github.com/feyninc/nobg

FeyNobg 是 feyn 在 BiRefNet（CAAI AIR 2024）基础上扩展的通用抠图模型：
Swin-Large 主干 + 三项自定义增强（梯度注意力 use_gradient_attention、
图像块注入 use_image_patch_injection、多尺度输入 use_multi_scale_input），
原生 1024×1024 推理，权重约 1.05GB。

与 SDMatte 的定位差异：
- FeyNobg 全自动，不需要任何提示，一步出 alpha，适合批量去背景；
- SDMatte 需要框/掩码提示来指定抠哪个目标，适合画面里有多个主体时精确取一个。

实现说明：上游 nobg 的预处理模块继承 transformers>=5.4 的 TorchvisionBackend，
在 ComfyUI 常见的 transformers 4.x 上会 ImportError，故 Ruinode 内嵌了推理子集
并重写了预处理（见 feynobg/birefnet/image_processing_birefnet.py），
数值规格与官方对齐，且不需要升级 transformers。
"""
import os

import numpy as np
import torch

import folder_paths

# ---------------------------------------------------------------- 模型目录注册

NOBG_DIR = os.path.join(folder_paths.models_dir, "nobg")
os.makedirs(NOBG_DIR, exist_ok=True)

HF_REPO = "feyninc/FeyNobg"
DEFAULT_MODEL_DIRNAME = "FeyNobg"

_MODEL_CACHE = {}

_NO_MODEL_HINT = "未找到模型，将自动下载"


def _is_model_dir(p):
    """一个模型快照目录需同时具备 config.json 与权重文件。"""
    if not os.path.isdir(p):
        return False
    if not os.path.isfile(os.path.join(p, "config.json")):
        return False
    return any(f.endswith((".safetensors", ".bin"))
               for f in os.listdir(p))


def _list_models():
    """扫描 models/nobg 下的模型快照目录。"""
    found = []
    if os.path.isdir(NOBG_DIR):
        for name in sorted(os.listdir(NOBG_DIR)):
            if _is_model_dir(os.path.join(NOBG_DIR, name)):
                found.append(name)
    return found or [_NO_MODEL_HINT]


def _ensure_model(name):
    """返回模型目录；不存在时从 HuggingFace 拉取到 models/nobg/FeyNobg。"""
    if name and name != _NO_MODEL_HINT:
        path = os.path.join(NOBG_DIR, name)
        if _is_model_dir(path):
            return path

    target = os.path.join(NOBG_DIR, DEFAULT_MODEL_DIRNAME)
    if _is_model_dir(target):
        return target

    print(f"[Ruinode-FeyNobg] 本地未找到模型，开始从 HuggingFace 下载 {HF_REPO}"
          f"（约 1.05GB）→ {target}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            HF_REPO, local_dir=target,
            allow_patterns=["*.json", "*.safetensors"],
        )
    except Exception as e:
        raise RuntimeError(
            f"自动下载失败：{e}\n"
            f"请手动下载 https://huggingface.co/{HF_REPO} 的 "
            f"config.json / preprocessor_config.json / model.safetensors，"
            f"放到：{target}"
        )
    if not _is_model_dir(target):
        raise RuntimeError(f"下载完成但目录不完整，请检查：{target}")
    print(f"[Ruinode-FeyNobg] 下载完成：{target}")
    return target


def _remap_swin_keys(state_dict):
    """
    把 transformers 5.x 保存的 Swin 权重键名，改写成 4.x 的命名。

    FeyNobg 的权重是用 transformers 5.x 导出的，而 ComfyUI 环境普遍还是 4.x，
    两版的 SwinBackbone 模块命名不同（数学结构一致，纯改名）：

        bb.swin.X                        -> bb.X                （5.x 多包一层 swin）
        attention.{q,k,v}_proj           -> attention.self.{query,key,value}
        attention.o_proj                 -> attention.output.dense
        attention.relative_position_bias
          .relative_position_bias_table  -> attention.self.relative_position_bias_table
        mlp.fc1 / mlp.fc2                -> intermediate.dense / output.dense

    不改名直接 load_state_dict(strict=False) 的后果：958 个参数里只有 405 个能对上，
    整个 backbone 形同随机初始化 —— 模型照样能跑完，但输出的 alpha 几乎全黑
    （实测 max 0.02、mean 0.000）。这类"不报错的错"最难排查，故此处显式处理。
    """
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("bb.swin."):
            nk = "bb." + nk[len("bb.swin."):]
        nk = nk.replace(".attention.q_proj.", ".attention.self.query.")
        nk = nk.replace(".attention.k_proj.", ".attention.self.key.")
        nk = nk.replace(".attention.v_proj.", ".attention.self.value.")
        nk = nk.replace(".attention.o_proj.", ".attention.output.dense.")
        nk = nk.replace(".attention.relative_position_bias.relative_position_bias_table",
                        ".attention.self.relative_position_bias_table")
        nk = nk.replace(".mlp.fc1.", ".intermediate.dense.")
        nk = nk.replace(".mlp.fc2.", ".output.dense.")
        out[nk] = v
    return out


def _load_weights_strict(model, ckpt_path):
    """加载权重并严格校验，宁可报错也不接受"能跑但结果劣化"的静默失配。"""
    from safetensors.torch import load_file

    sd = load_file(ckpt_path)
    own = set(model.state_dict().keys())
    direct = len(own & set(sd.keys()))
    remapped = _remap_swin_keys(sd)
    after = len(own & set(remapped.keys()))

    # 权重与本机 transformers 命名不一致时才改写；若环境本就是 5.x，直接匹配更优
    if after > direct:
        print(f"[Ruinode-FeyNobg] 权重键名按 transformers 4.x 改写："
              f"{direct} -> {after} / {len(own)} 匹配")
        sd = remapped

    result = model.load_state_dict(sd, strict=False)

    # relative_position_index 是由窗口大小推出的确定性 buffer，构造时已算好，不需要权重；
    # bb.layernorm 是 backbone 末端的 norm，BiRefNet 只取中间各级特征图，用不到。
    # 这两类之外的任何缺失/多余，都意味着结构对不上，必须叫停。
    bad_missing = [k for k in result.missing_keys
                   if not k.endswith("relative_position_index")]
    bad_unexpected = [k for k in result.unexpected_keys
                      if not k.startswith("bb.layernorm.")]
    if bad_missing or bad_unexpected:
        for k in bad_missing[:8]:
            print(f"    未被覆盖: {k}")
        for k in bad_unexpected[:8]:
            print(f"    未被使用: {k}")
        raise RuntimeError(
            f"权重与网络结构不匹配（异常缺失 {len(bad_missing)}，异常多余 "
            f"{len(bad_unexpected)}）。继续推理只会得到劣化结果，故中止。\n"
            f"请确认权重为 feyninc/FeyNobg，且 transformers 版本受支持"
            f"（已在 4.57 与 5.x 上验证）。"
        )
    print(f"[Ruinode-FeyNobg] 权重加载完成（{len(own) - len(result.missing_keys)}"
          f"/{len(own)} 个张量，另 {len(result.missing_keys)} 个为确定性 buffer）")


def _load_model(model_dir, dtype, device):
    key = (model_dir, str(dtype), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    import dataclasses
    import json

    from .feynobg import BiRefNet, BiRefNetConfig

    print(f"[Ruinode-FeyNobg] 加载模型：{model_dir}（{dtype}，{device}）")
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    # config.json 里含 nobg_version 等非模型字段，按 dataclass 的字段名过滤
    valid = {f.name for f in dataclasses.fields(BiRefNetConfig)}
    cfg = BiRefNetConfig(**{k: v for k, v in raw_cfg.items() if k in valid})

    ckpt = os.path.join(model_dir, "model.safetensors")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"缺少权重文件：{ckpt}")

    _MODEL_CACHE.clear()  # 单份约 1GB，不做多份缓存
    # 不走 BiRefNet.from_pretrained：它内部按 huggingface_hub 的宽松策略加载，
    # 键名对不上时会静默丢弃权重（正是上面 _remap_swin_keys 说明的那个坑）
    model = BiRefNet(cfg)
    _load_weights_strict(model, ckpt)
    model.eval()
    model.to(device=device, dtype=dtype)
    _MODEL_CACHE[key] = model
    return model


class RuiFeyNobg:
    """FeyNobg 全自动抠图：输入图像，输出 alpha 与去背景图。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (_list_models(), {
                    "tooltip": "放在 models/nobg 下的模型目录（需含 config.json 与 "
                               "model.safetensors）。\n"
                               "留空或未找到时，首次运行会自动从 HuggingFace 下载 "
                               "feyninc/FeyNobg（约 1.05GB）。"
                }),
                "resolution": ([512, 768, 1024, 1280, 1536], {
                    "default": 1024,
                    "tooltip": "推理分辨率。模型原生训练分辨率为 1024，改动会影响细节与显存：\n"
                               "调低更省显存但边缘变粗；调高不一定更好，可能出现结构断裂。"
                }),
                "precision": (["fp32", "fp16"], {
                    "default": "fp32",
                    "tooltip": "实测两者输出一致（同图 alpha 均值都是 0.657），\n"
                               "但 fp16 显存减半且明显更快，推荐优先用 fp16。\n"
                               "保留 fp32 作默认，是为老显卡上半精度异常时有个退路。"
                }),
                "device": (["auto", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "alpha_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "多大置信度才算前景。模型对拿不准的区域会输出 0.5 上下的\n"
                               "中间值，表现为「整片主体半透明发灰」。\n"
                               "把阈值调低（如 0.3）可把这类区域拉回不透明。\n"
                               "注意：只对已有一定响应的区域有效；模型压根没认出来的\n"
                               "地方 alpha 接近 0，再降阈值也救不回来。"
                }),
                "alpha_softness": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "阈值两侧过渡带的宽度，决定边缘软硬。\n"
                               "1.0 = 完全不处理，原样输出模型结果（默认）\n"
                               "0.2~0.4 = 压掉灰雾但保留发丝级过渡（推荐从 0.3 试）\n"
                               "0 = 硬二值化，边缘变成锯齿硬边，抠玻璃/头发慎用"
                }),
                "keep_aspect_ratio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "模型固定吃 1024×1024，默认会把图直接拉伸成正方形\n"
                               "（与官方训练方式一致）。长图/宽图形变严重时可开启此项，\n"
                               "改为等比缩放 + 边缘延展补边，推理后再裁掉补边部分。\n"
                               "注意这与训练分布不同，属于试验性选项：\n"
                               "极端长宽比（如手机截图 1:2 以上）通常有改善，常规比例建议关闭。"
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
        """模型列表随目录变化，宽松放行，运行时兜底（含自动下载）。"""
        return True

    @staticmethod
    def _remap_alpha(alpha, threshold, softness):
        """
        按阈值 / 柔和度重新映射 alpha，相当于给遮罩拉一次色阶。

        以 threshold 为中心、softness 为宽度取一段区间线性拉伸到 [0,1]：
        区间以下压成全透明，以上提成全不透明，区间内保留平滑过渡。
        默认 (0.5, 1.0) 时区间恰好是 [0,1]，等于原样返回。
        """
        low = threshold - softness / 2.0
        high = threshold + softness / 2.0
        if high <= low:                      # softness=0：硬二值化
            return (alpha >= threshold).to(alpha.dtype)
        if abs(low) < 1e-6 and abs(high - 1.0) < 1e-6:
            return alpha                     # 默认参数，一个像素都不动
        return ((alpha - low) / (high - low)).clamp(0.0, 1.0)

    def matting(self, image, model_name, resolution, precision, device,
                alpha_threshold=0.5, alpha_softness=1.0,
                keep_aspect_ratio=False, invert_mask=False):
        import comfy.model_management

        from .feynobg import BiRefNetImageProcessor

        model_dir = _ensure_model(model_name)
        dev = torch.device("cpu") if device == "cpu" \
            else comfy.model_management.get_torch_device()
        dtype = torch.float32 if precision == "fp32" else torch.float16
        model = _load_model(model_dir, dtype, dev)

        proc = BiRefNetImageProcessor.from_pretrained(model_dir)
        res = int(resolution)
        proc.size = {"height": res, "width": res}

        # ComfyUI 的 IMAGE 约定是 RGB，但上游抠图类节点常输出 RGBA
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
            # 逐张推理：1024 分辨率下 Swin-Large 峰值显存不低，整批一次容易 OOM
            pixel_values, meta = proc.preprocess_tensor(
                image[i:i + 1], device=dev, dtype=dtype,
                keep_aspect=bool(keep_aspect_ratio))
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
            # fp16 推理出的 logits 先转回 fp32 再 sigmoid/缩放，避免精度损失
            if isinstance(outputs, dict):
                outputs = {"logits": outputs["logits"].float()}
            alpha = proc.post_process_alpha_matting(
                outputs, target_sizes=[(H, W)], crop=meta)[0]
            alphas.append(alpha.clamp(0, 1).cpu())

        alpha = torch.stack(alphas)  # [B,H,W]
        alpha = self._remap_alpha(alpha, float(alpha_threshold),
                                  float(alpha_softness))
        if invert_mask:
            alpha = 1.0 - alpha

        cutout = image.detach().cpu().float() * alpha.unsqueeze(-1)
        return (alpha, cutout)


NODE_CLASS_MAPPINGS = {
    "RuiFeyNobg": RuiFeyNobg,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiFeyNobg": "FeyNobg 抠图 / FeyNobg Matting",
}
