# -*- coding: utf-8 -*-
"""
半透明抠图节点（Unmult）—— Ruinode
==================================
纯数学去底算法，等效 AE Unmult 效果。
适用于渲染在纯色底上的光效、火焰、烟雾、粒子、UI 特效等半透明素材。

原理：纯色背景合成图满足 C = αF + (1-α)B，
其中 B 为已知背景色。通过各通道与背景色的差异反推 α 和前景色 F。
支持任意背景色（黑/白/绿幕/自定义），纯数学变换，无模型推理。

AI 主体保护（可选）：接入 BiRefNet / FeyNobg 等抠图节点输出的 subject_mask，
用 max(unmult_α, subject_mask) 合并，防止主体中与背景色相近的区域被误判为透明。
该合并由「主体保护」开关控制，关闭时即便连了 mask 也不采纳。
典型场景：黑底光效中人物穿黑衣 → 纯 Unmult 会让黑衣半透明，
接入主体 mask 后黑衣区域强制保留为不透明。

支持批量输入（序列帧/视频帧）。批量走 torch 多线程分块运算，
结果直写预分配张量；124 帧 1024x1024 实测 2.6 秒。
"""

import numpy as np
import torch


def _hex_to_rgb01(hex_str: str) -> tuple:
    """将 #RRGGBB 格式的颜色字符串转为 (r, g, b) 浮点元组，值域 [0,1]。"""
    h = hex_str.strip().lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    if len(h) != 6:
        raise ValueError(f"无效的颜色格式：{hex_str}，需要 #RRGGBB")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def _unmult_frame(rgb: np.ndarray, bg_color: tuple,
                  alpha_low: float, alpha_high: float,
                  subject_mask: np.ndarray = None,
                  epsilon: float = 1e-6) -> tuple:
    """对单帧图像执行 Unmult 去底，可选主体保护。

    注意：这是单帧参考实现，节点主路径已改走 unmult() 里的 torch 分块
    批处理（快约 3.8 倍）。此函数保留用于对拍验证和外部复用，
    两者数值逐元素一致。

    参数:
        rgb: float32 [H,W,3] 值域 [0,1]
        bg_color: (R,G,B) 值域 [0,1]
        alpha_low: 黑点（低于此值的 alpha 映射为 0）
        alpha_high: 白点（高于此值的 alpha 映射为 1）
        subject_mask: float32 [H,W] 值域 [0,1]，主体区域为 1

    返回:
        (foreground [H,W,3], alpha [H,W])  均 float32
    """
    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    diff = rgb.astype(np.float32) - bg

    scale = np.array([max(bg_color[c], 1.0 - bg_color[c], epsilon)
                      for c in range(3)], dtype=np.float32).reshape(1, 1, 3)
    norm_diff = np.abs(diff) / scale
    alpha = np.max(norm_diff, axis=-1)
    alpha = np.clip(alpha, 0.0, 1.0)

    if alpha_low > 0.0 or alpha_high < 1.0:
        span = max(alpha_high - alpha_low, epsilon)
        alpha = np.clip((alpha - alpha_low) / span, 0.0, 1.0)

    if subject_mask is not None:
        alpha = np.maximum(alpha, subject_mask)

    alpha_safe = np.maximum(alpha, epsilon)
    foreground = bg + diff / alpha_safe[..., np.newaxis]
    foreground = np.clip(foreground, 0.0, 1.0)

    transparent = alpha < epsilon
    foreground[transparent] = 0.0

    return foreground.astype(np.float32), alpha.astype(np.float32)


class RuiUnmult:
    """半透明抠图（Unmult）：指定背景色，纯数学去底，输出 RGBA 图像 + Alpha 蒙版。
    可选接入 AI 主体保护 mask，防止主体中近似背景色的区域被误判为透明。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "待去底的图像，支持批量（序列帧/视频帧逐帧处理）。"
                }),
                "bg_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "要去除的背景色，#RRGGBB 格式。\n"
                               "常用值：#000000（黑底）、#FFFFFF（白底）、"
                               "#00FF00（绿幕）、#FF00FF（品红）。"
                }),
                "黑点": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "display": "slider",
                    "tooltip": "低于此值的 alpha 强制归零。\n"
                               "调高可清除背景残留噪点，但过高会丢失边缘细节。"
                }),
                "白点": ("FLOAT", {
                    "default": 1.0, "min": 0.3, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "高于此值的 alpha 强制归一。\n"
                               "调低可让主体更实、减少半透明损失，但过低会让边缘硬化。"
                }),
                "主体保护": ("BOOLEAN", {
                    "default": True,
                    "label_on": "启用",
                    "label_off": "关闭",
                    "tooltip": "是否采纳下方接入的 subject_mask。\n\n"
                               "启用：alpha 取 max(unmult 结果, subject_mask)，\n"
                               "  主体区域强制不透明，光效边缘仍保留半透明。\n"
                               "  黑底素材里的黑衣、黑发全靠它保住。\n\n"
                               "关闭：即便已经连了 subject_mask 也完全不采纳，\n"
                               "  等同于纯 Unmult。想对比「有无 AI 介入」的差别时，\n"
                               "  拨这个开关即可，不必拔线。"
                }),
            },
            "optional": {
                "subject_mask": ("MASK", {
                    "tooltip": "AI 主体保护遮罩（可选）。\n"
                               "接入 FeyNobg / Lucida / BiRefNet 等抠图节点输出的 alpha，\n"
                               "节点会执行 max(unmult_α, subject_mask) 合并：\n"
                               "主体内部强制不透明，光效边缘保留半透明。\n\n"
                               "典型场景：黑底光效中人物穿黑衣 →\n"
                               "纯 Unmult 会让黑衣半透明，接入主体 mask 后黑衣保留。\n\n"
                               "不连接、或上方「主体保护」关闭时，等同于纯 Unmult，无 AI 介入。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba_image", "alpha")
    FUNCTION = "unmult"
    CATEGORY = "Rui-Node🐶/抠图✂️"

    def unmult(self, **kwargs):
        image = kwargs.get("image")
        bg_color = kwargs.get("bg_color", "#000000")
        alpha_low = kwargs.get("黑点", kwargs.get("alpha_low", 0.0))
        alpha_high = kwargs.get("白点", kwargs.get("alpha_high", 1.0))
        # 主体保护开关：关闭时彻底忽略 subject_mask，哪怕上游已经连线
        use_subject = kwargs.get("主体保护", kwargs.get("use_subject_mask", True))
        subject_mask_tensor = kwargs.get("subject_mask", None)
        if not use_subject:
            subject_mask_tensor = None

        bg_rgb = _hex_to_rgb01(bg_color)

        B, H, W, C = image.shape
        if C == 4:
            image = image[..., :3]
        elif C == 1:
            image = image.repeat(1, 1, 1, 3)

        # 主体 mask 预处理：统一成 [N,H,W] float32，只做一次。
        # 原先是循环里逐帧 resize，mask 只有一帧时会被重复 resize B 次。
        smask_all = None
        if subject_mask_tensor is not None:
            sm = subject_mask_tensor
            if sm.dim() == 2:
                sm = sm.unsqueeze(0)
            sm = sm.float()
            if sm.shape[1] != H or sm.shape[2] != W:
                import cv2
                arr = sm.cpu().numpy()
                arr = np.stack([cv2.resize(a, (W, H),
                                           interpolation=cv2.INTER_LINEAR)
                                for a in arr])
                sm = torch.from_numpy(arr)
            smask_all = sm.clamp(0.0, 1.0)

        # ── 批处理 ──
        # 原实现是「逐帧 numpy → 收集成 list → torch.stack」。实测 124 帧
        # 1024×1024 耗时 9 秒，瓶颈有二：单帧内近十次中间数组分配（每份
        # 12MB），以及 numpy 绝大多数算子单线程、多核完全闲置。
        # 改为 torch 分块处理：torch 的 CPU 算子走多线程，原地运算压掉中间
        # 分配，结果直接写入预分配张量、省掉最后那次 stack 大拷贝。
        # 分块而不是一口气吃下整批，是因为整批中间量会膨胀到数 GB。
        px = max(1, H * W)
        chunk = max(1, min(B, int(24_000_000 // px)))

        bg_t = torch.tensor(bg_rgb, dtype=torch.float32).view(1, 1, 1, 3)
        scale_t = torch.tensor(
            [max(bg_rgb[c], 1.0 - bg_rgb[c], 1e-6) for c in range(3)],
            dtype=torch.float32).view(1, 1, 1, 3)
        eps = 1e-6
        lo, hi = float(alpha_low), float(alpha_high)
        use_levels = (lo > 0.0) or (hi < 1.0)
        span = max(hi - lo, eps)

        rgba_out = torch.empty((B, H, W, 4), dtype=torch.float32)
        alpha_out = torch.empty((B, H, W), dtype=torch.float32)

        for s in range(0, B, chunk):
            e = min(B, s + chunk)
            blk = image[s:e]
            if blk.dtype != torch.float32:
                blk = blk.float()

            diff = blk - bg_t                                    # [b,H,W,3]
            alpha = diff.abs().div_(scale_t).amax(dim=-1).clamp_(0.0, 1.0)

            if use_levels:
                alpha.sub_(lo).div_(span).clamp_(0.0, 1.0)

            if smask_all is not None:
                # mask 帧数不足时复用最后一帧，与逐帧版一致
                idx = torch.arange(s, e).clamp_(max=smask_all.shape[0] - 1)
                alpha = torch.maximum(alpha, smask_all[idx])

            # 反解前景：C = αF + (1-α)B  ⇒  F = B + (C-B)/α
            fg = diff.div_(alpha.clamp(min=eps).unsqueeze(-1)).add_(bg_t)
            fg.clamp_(0.0, 1.0)
            fg.mul_((alpha >= eps).unsqueeze(-1))    # 全透明处前景归零

            rgba_out[s:e, :, :, :3] = fg
            rgba_out[s:e, :, :, 3] = alpha
            alpha_out[s:e] = alpha

        return (rgba_out, alpha_out)


NODE_CLASS_MAPPINGS = {
    "RuiUnmult": RuiUnmult,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiUnmult": "半透明抠图 / Unmult Matting",
}
