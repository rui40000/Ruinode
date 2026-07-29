# -*- coding: utf-8 -*-
"""
BiRefNet 预处理 / 后处理（Ruinode 内嵌版）
==========================================
官方 nobg 的同名文件继承 transformers>=5.4 的 TorchvisionBackend，而 ComfyUI
常见环境仍是 transformers 4.x（本机 T8 为 4.49.0），直接引入会报：

    ModuleNotFoundError: No module named 'transformers.image_processing_backends'

升级 transformers 到 5.x 极可能连累其他自定义节点，因此这里用纯 torch/PIL
复刻官方的同一套数值规格，去掉对 transformers 的依赖：

- 预处理：转 RGB → 双线性缩放到 size（抗锯齿）→ 归一到 [0,1] → ImageNet 标准化
- 后处理：logits.sigmoid() → 双线性缩放回原尺寸
  （**先 sigmoid 再缩放**，与官方 post_process_alpha_matting 及其 eval 脚本一致，
    顺序颠倒会让边缘过渡发生偏移）
- cutout：把 alpha 写入原图的 alpha 通道，输出 RGBA

除去掉 transformers 依赖外，逐项对齐官方实现；训练用的 segmentation_maps /
labels 分支推理不需要，未移植。
"""
import json
import os

import torch
import torch.nn.functional as F

# 与 transformers.image_utils 的 IMAGENET_DEFAULT_MEAN / STD 相同，
# 也与 FeyNobg 的 preprocessor_config.json 一致
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class BiRefNetImageProcessor:
    """BiRefNet 图像处理器（预处理 + 后处理）。"""

    def __init__(self, size=None, image_mean=None, image_std=None,
                 rescale_factor=1 / 255, **kwargs):
        size = size or {"height": 1024, "width": 1024}
        self.size = {
            "height": int(size.get("height", 1024)),
            "width": int(size.get("width", 1024)),
        }
        self.image_mean = tuple(image_mean or IMAGENET_DEFAULT_MEAN)
        self.image_std = tuple(image_std or IMAGENET_DEFAULT_STD)
        self.rescale_factor = rescale_factor

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """从本地目录的 preprocessor_config.json 读取配置（不联网）。"""
        cfg = {}
        if os.path.isdir(pretrained_model_name_or_path):
            p = os.path.join(pretrained_model_name_or_path,
                             "preprocessor_config.json")
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
        cfg.update(kwargs)
        return cls(
            size=cfg.get("size"),
            image_mean=cfg.get("image_mean"),
            image_std=cfg.get("image_std"),
            rescale_factor=cfg.get("rescale_factor", 1 / 255),
        )

    # ---------------------------------------------------------------- 预处理
    def preprocess_tensor(self, images, device=None, dtype=torch.float32,
                          keep_aspect=False):
        """
        ComfyUI 原生张量入口，避免 PIL 往返。

        images: (B, H, W, 3) float，值域 [0,1]（ComfyUI 的 IMAGE 约定）
        keep_aspect: False 走官方路径，直接拉伸成正方形（与训练一致）；
                     True 则按长边等比缩放后补边，保留原始比例。

        返回 (pixel_values, meta):
            pixel_values (B, 3, size_h, size_w)，已 ImageNet 标准化
            meta         keep_aspect 时为有效区域 (nh, nw)，供后处理裁掉补边；否则 None
        """
        x = images.permute(0, 3, 1, 2).contiguous().float()  # BHWC -> BCHW
        th, tw = self.size["height"], self.size["width"]
        meta = None

        if keep_aspect:
            _, _, H, W = x.shape
            scale = min(th / H, tw / W)
            nh = max(1, min(th, int(round(H * scale))))
            nw = max(1, min(tw, int(round(W * scale))))
            x = F.interpolate(x, size=(nh, nw), mode="bilinear",
                              align_corners=False, antialias=True)
            # 补边放在右/下侧，用边缘像素延展而非填黑：填黑会凭空造出一条
            # 高对比直边，模型容易把它当成物体轮廓
            if nh < th or nw < tw:
                x = F.pad(x, (0, tw - nw, 0, th - nh), mode="replicate")
            meta = (nh, nw)
        else:
            # 官方走 torchvision 的 resize，对张量默认开抗锯齿；缩小到 1024 时
            # 是否抗锯齿对细边缘影响可见，这里保持一致
            x = F.interpolate(x, size=(th, tw), mode="bilinear",
                              align_corners=False, antialias=True)

        mean = torch.tensor(self.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        x = (x - mean) / std
        x = x.to(device=device, dtype=dtype) if device is not None \
            else x.to(dtype=dtype)
        return x, meta

    def __call__(self, images, return_tensors="pt", **kwargs):
        """PIL 入口，兼容官方 README 的用法：processor(image, return_tensors='pt')。"""
        import numpy as np
        from PIL import Image as PILImage

        if isinstance(images, PILImage.Image):
            images = [images]
        arrs = []
        for im in images:
            im = im.convert("RGB")
            arrs.append(np.asarray(im, dtype=np.float32) * self.rescale_factor)
        batch = torch.from_numpy(np.stack(arrs))  # (B,H,W,3) in [0,1]
        return {"pixel_values": self.preprocess_tensor(batch)[0]}

    # ---------------------------------------------------------------- 后处理
    def post_process_alpha_matting(self, outputs, target_sizes=None, crop=None):
        """
        原始 logits -> 每图 alpha（[0,1]，形状 (H, W)）。

        outputs: 含 "logits" 的 dict 或带 .logits 的对象，logits 形状 (B,1,H,W)
        target_sizes: [(h, w), ...]，逐图缩放回原尺寸
        crop: 预处理若做过等比补边，这里传 (nh, nw) 把补出来的部分裁掉
        """
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        if crop is not None:
            nh, nw = crop
            logits = logits[:, :, :nh, :nw]
        if target_sizes is not None and len(logits) != len(target_sizes):
            raise ValueError(
                f"给了 {len(target_sizes)} 个目标尺寸，但批次里有 {len(logits)} 张图"
            )
        # 官方注释明确：先 sigmoid 再缩放，与 eval/benchmark 脚本一致
        probs = logits.sigmoid()
        mattes = []
        for idx in range(len(probs)):
            alpha = probs[idx].unsqueeze(0)  # (1,1,H,W)
            if target_sizes is not None:
                alpha = F.interpolate(
                    alpha, size=target_sizes[idx],
                    mode="bilinear", align_corners=False,
                )
            mattes.append(alpha[0, 0])
        return mattes

    @staticmethod
    def cutout(image, alpha):
        """把 alpha 合成到原图，返回 RGBA。alpha 可为 (H,W) 张量或 PIL 图。"""
        from PIL import Image as PILImage

        if isinstance(alpha, torch.Tensor):
            arr = (alpha.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            alpha = PILImage.fromarray(arr, mode="L")
        if alpha.size != image.size:
            alpha = alpha.resize(image.size, PILImage.Resampling.BILINEAR)
        cutout = image.convert("RGBA")
        cutout.putalpha(alpha)
        return cutout
