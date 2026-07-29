# -*- coding: utf-8 -*-
"""
像素化节点（Ruinode）
=====================
把普通图像转成**能直接当素材用的像素画**，而不是"马赛克滤镜"。
两者的区别在于：滤镜只是把画面涂成方块，输出仍是原尺寸的大图；
而像素游戏要的是真实小分辨率、颜色数受控、边缘硬朗的 sprite。

三种工作模式：
- 按目标宽度   给定输出宽度（如 64），普通图/照片/插画 → 像素画
- 按像素块大小 每 N×N 原像素合成一个像素，适合已知放大倍数时精确还原
- 自动检测网格 探测图像本身隐含的像素网格并还原
  —— 专治「AI 生成的伪像素图」：模型输出的 1024×1024 图看着像素风，
     实际网格歪斜、边缘带抗锯齿、颜色成千上万，塞进引擎会糊。

关于方案选型（研究后的结论）：
Sprite Fusion 的 Pixel Snapper（MIT）解决的是「伪像素图 → 完美像素图」，
思路是检测网格 + 按主导色重采样；而「普通图 → 像素画」是另一个问题，
核心在降采样方式与调色板量化。本节点把两条路都做进同一个节点，
算法为纯 numpy/PIL 自研实现（无额外依赖、无需模型权重），其中网格检测
按公开研究的要点处理了两类经典误判：谐波（八度）错误与内容周期冒充像素周期，
详见 pixelart/grid.py。
"""
import numpy as np
import torch

from .pixelart import (PALETTE_NAMES, PALETTES, apply_palette, detect_grid,
                       downsample_grid, kmeans_palette, median_cut_palette)

_MODES = ["按目标宽度", "按像素块大小", "自动检测网格（AI伪像素图还原）"]

_DOWN = {
    "主导色 dominant（像素画首选）": "dominant",
    "中位数 median": "median",
    "均值 mean（会糊边，慎用）": "mean",
    "中心像素 center（最锐）": "center",
}

_DITHER = {
    "无（默认）": "none",
    "Bayer 2×2": "bayer2",
    "Bayer 4×4": "bayer4",
    "Bayer 8×8": "bayer8",
    "Floyd-Steinberg": "floyd-steinberg",
    "随机噪声": "noise",
}

_PAL_ADAPTIVE = ["自适应 k-means（质量优先）", "自适应 median cut（速度优先）"]
_PAL_OPTIONS = ["不量化"] + _PAL_ADAPTIVE + PALETTE_NAMES


def _nearest_resize(img, tw, th):
    """最近邻缩放。绝不插值 —— 插值会造出调色板外的新颜色并糊掉硬边。"""
    h, w = img.shape[:2]
    if (w, h) == (tw, th):
        return img
    xi = np.clip((np.arange(tw) * (w / tw)).astype(np.int64), 0, w - 1)
    yi = np.clip((np.arange(th) * (h / th)).astype(np.int64), 0, h - 1)
    return img[yi][:, xi]


class RuiPixelate:
    """图像 → 像素画（面向像素游戏资产）。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "待像素化的图像。批量输入时若各图比例不同，\n"
                               "会统一到首张的输出尺寸（ComfyUI 的 IMAGE 必须同形状）。"
                }),
                "mode": (_MODES, {
                    "default": "按目标宽度",
                    "tooltip": "按目标宽度：普通图转像素画，直接给输出宽度\n"
                               "按像素块大小：每 N×N 原像素合成一个像素\n"
                               "自动检测网格：探测图中隐含的像素网格并还原，\n"
                               "  专治 AI 生成的模糊伪像素图（网格歪、带抗锯齿）"
                }),
                "target_width": ("INT", {
                    "default": 64, "min": 8, "max": 2048, "step": 1,
                    "tooltip": "仅「按目标宽度」模式生效。高度按原图比例自动计算。\n"
                               "常见 sprite 尺寸：16 / 32 / 48 / 64 / 96 / 128"
                }),
                "pixel_size": ("INT", {
                    "default": 8, "min": 1, "max": 256, "step": 1,
                    "tooltip": "仅「按像素块大小」模式生效：每 N×N 原像素 → 1 像素。"
                }),
                "downsample": (list(_DOWN.keys()), {
                    "default": "主导色 dominant（像素画首选）",
                    "tooltip": "每个单元如何定色。\n"
                               "主导色：取块内出现最多的颜色，不会凭空造出新颜色（首选）\n"
                               "中位数：抗噪，偶尔比主导色更稳\n"
                               "均值：会产生新颜色并糊边，只在想要柔和过渡时用\n"
                               "中心像素：等价最近邻，最锐利但受噪点影响"
                }),
                "palette": (_PAL_OPTIONS, {
                    "default": "自适应 k-means（质量优先）",
                    "tooltip": "颜色数受控是像素画风格的一部分，也方便整套素材统一改色。\n"
                               "自适应：从画面自身聚类出调色板（k-means 在 CIELAB 空间，\n"
                               "  比 RGB 更贴合人眼，暗部层次保留更好）\n"
                               "固定盘：PICO-8 / Game Boy 等复古机型的真实调色板"
                }),
                "palette_size": ("INT", {
                    "default": 16, "min": 2, "max": 256, "step": 1,
                    "tooltip": "仅自适应调色板生效。选固定调色板时其颜色数已定，此项忽略。\n"
                               "参考：8~16 复古感强，32~64 细节保留更多。"
                }),
                "dither": (list(_DITHER.keys()), {
                    "default": "无（默认）",
                    "tooltip": "颜色数很少时用抖动能换回一些层次，代价是引入噪点。\n"
                               "Bayer：规则网点，复古感强、可平铺，像素画最常用\n"
                               "Floyd-Steinberg：过渡最自然，但纹理不规则、不利于后期手改\n"
                               "手绘风像素画通常不抖动，先试「无」。"
                }),
                "output_scale": ("INT", {
                    "default": 1, "min": 1, "max": 32, "step": 1,
                    "tooltip": "输出放大倍数（最近邻，不插值）。\n"
                               "1 = 真实像素尺寸，直接可用作 sprite（推荐）\n"
                               ">1 仅为了在 ComfyUI 里看清效果，导出素材前记得改回 1"
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "可选。抠图得到的 alpha 接进来会按同一网格降采样，\n"
                               "并按 mask_threshold 二值化成硬边 —— sprite 需要硬边 alpha。"
                }),
                "dither_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05,
                    "tooltip": "抖动幅度。已按调色板的平均色距归一化，\n"
                               "所以同一数值在 4 色盘和 64 色盘上观感接近。"
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "遮罩二值化阈值，高于它算不透明。\n"
                               "设为 0 则保留灰度遮罩（不二值化）。"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xFFFFFFFF,
                    "tooltip": "随机种子。影响 k-means 调色板的初始化与随机噪声抖动；\n"
                               "同一张图换种子会得到略有差异的配色，可多试几个挑顺眼的。\n"
                               "固定调色板与 Bayer 抖动不受它影响。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "info")
    FUNCTION = "pixelate"
    CATEGORY = "Rui-Node🐶/图像调节🎨"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    # ---------------------------------------------------------- 单张处理
    def _one(self, img, msk, mode, target_width, pixel_size, down, pal_opt,
             pal_size, dither, strength, mask_th, seed, notes):
        H, W = img.shape[:2]

        # ---- 1. 确定网格并降采样 ----
        if mode.startswith("自动检测"):
            info = detect_grid(img)
            sx, sy = info["size_x"], info["size_y"]
            ox, oy = info["off_x"], info["off_y"]
            if info["detected"]:
                notes.append(
                    f"检测到网格 {sx}×{sy}（相位 {ox},{oy}；置信度 "
                    f"{info['conf_x']:.1f}/{info['conf_y']:.1f}）")
            else:
                notes.append(
                    f"未检出可靠网格（置信度 {info['conf_x']:.1f}/"
                    f"{info['conf_y']:.1f}，低于阈值）—— 该图可能本就不是"
                    f"放大的像素图。已按检测到的 {sx}×{sy} 处理，"
                    f"建议改用「按目标宽度」模式")
            small = downsample_grid(img, sx, sy, ox, oy, down)
            small_m = downsample_grid(msk[..., None], sx, sy, ox, oy,
                                      "mean")[..., 0] if msk is not None else None
        elif mode.startswith("按像素块"):
            s = int(pixel_size)
            small = downsample_grid(img, s, s, 0, 0, down)
            small_m = downsample_grid(msk[..., None], s, s, 0, 0,
                                      "mean")[..., 0] if msk is not None else None
            notes.append(f"块大小 {s}×{s}")
        else:
            ow = int(max(1, min(target_width, W)))
            oh = max(1, int(round(H * ow / W)))
            # 先最近邻对齐到整数倍，再走等距网格：这样输出尺寸精确，
            # 又不会像先做面积平均那样把颜色糊掉
            k = max(1, int(round(W / ow)))
            im2 = _nearest_resize(img, ow * k, oh * k)
            small = downsample_grid(im2, k, k, 0, 0, down)
            if msk is not None:
                m2 = _nearest_resize(msk[..., None], ow * k, oh * k)
                small_m = downsample_grid(m2, k, k, 0, 0, "mean")[..., 0]
            else:
                small_m = None
            notes.append(f"目标宽度 {ow} → 输出 {small.shape[1]}×{small.shape[0]}")

        # ---- 2. 调色板量化 ----
        if pal_opt != "不量化":
            if pal_opt in PALETTES:
                pal = PALETTES[pal_opt] / 255.0
            elif pal_opt.startswith("自适应 k-means"):
                pal = kmeans_palette(small, int(pal_size), seed=int(seed))
            else:
                pal = median_cut_palette(small, int(pal_size))
            small = apply_palette(small, pal, _DITHER.get(dither, "none"),
                                  float(strength), int(seed))
            notes.append(f"调色板 {pal_opt}（{pal.shape[0]} 色）"
                         + (f" + {dither}" if _DITHER.get(dither) != "none" else ""))
        else:
            uniq = np.unique(
                (np.clip(small, 0, 1) * 255).astype(np.uint8).reshape(-1, 3),
                axis=0).shape[0]
            notes.append(f"未量化（实际 {uniq} 色）")

        if small_m is not None and mask_th > 0:
            small_m = (small_m >= float(mask_th)).astype(np.float32)

        return np.clip(small, 0.0, 1.0), small_m

    def pixelate(self, image, mode, target_width, pixel_size, downsample,
                 palette, palette_size, dither, output_scale,
                 mask=None, dither_strength=1.0, mask_threshold=0.5, seed=0):
        down = _DOWN.get(downsample, "dominant")
        B, H, W, C = image.shape
        if C == 4:
            image = image[..., :3]
        elif C == 1:
            image = image.repeat(1, 1, 1, 3)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] != B:
                mask = mask[:1].repeat(B, 1, 1)

        outs, masks, notes = [], [], []
        for b in range(B):
            img = image[b].detach().cpu().float().numpy()
            msk = mask[b].detach().cpu().float().numpy() if mask is not None else None
            if msk is not None and msk.shape[:2] != img.shape[:2]:
                msk = _nearest_resize(msk[..., None], img.shape[1],
                                      img.shape[0])[..., 0]
            n = []
            small, small_m = self._one(
                img, msk, mode, target_width, pixel_size, down, palette,
                palette_size, dither, dither_strength, mask_threshold,
                seed + b, n)
            if b == 0:
                notes = n
            k = int(max(1, output_scale))
            if k > 1:
                small = np.repeat(np.repeat(small, k, axis=0), k, axis=1)
                if small_m is not None:
                    small_m = np.repeat(np.repeat(small_m, k, axis=0), k, axis=1)
            outs.append(torch.from_numpy(np.ascontiguousarray(small)))
            masks.append(torch.from_numpy(np.ascontiguousarray(
                small_m if small_m is not None
                else np.ones(small.shape[:2], dtype=np.float32))))

        # 批内各图尺寸可能不同（原图比例不一），此时只能退回逐张，
        # 但 ComfyUI 的 IMAGE 必须是同形状张量，故统一到首张尺寸
        h0, w0 = outs[0].shape[:2]
        for i in range(1, len(outs)):
            if outs[i].shape[:2] != (h0, w0):
                outs[i] = torch.from_numpy(_nearest_resize(
                    outs[i].numpy(), w0, h0))
                masks[i] = torch.from_numpy(_nearest_resize(
                    masks[i].numpy()[..., None], w0, h0)[..., 0])

        img_out = torch.stack(outs)
        msk_out = torch.stack(masks)
        info = f"{W}×{H} → {w0}×{h0}" + (f"（预览放大 {output_scale}×）"
                                          if output_scale > 1 else "")
        info += "\n" + "\n".join(notes)
        print(f"[Ruinode-Pixelate] {info}")
        return (img_out, msk_out, info)


NODE_CLASS_MAPPINGS = {
    "RuiPixelate": RuiPixelate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiPixelate": "像素化 / Pixelate",
}
