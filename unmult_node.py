# -*- coding: utf-8 -*-
"""
半透明抠图节点（Unmult）—— Ruinode
==================================
纯数学去底算法，等效 AE Unmult 效果。
适用于渲染在纯色底上的光效、火焰、烟雾、粒子、UI 特效等半透明素材。

原理：纯色背景合成图满足 C = αF + (1-α)B，
其中 B 为已知背景色。通过各通道与背景色的差异反推 α 和前景色 F。
支持任意背景色（黑/白/绿幕/自定义），纯数学变换，无模型推理。

支持批量输入（序列帧/视频帧），逐帧处理后堆叠输出。
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
                  epsilon: float = 1e-6) -> tuple:
    """对单帧图像执行 Unmult 去底。

    参数:
        rgb: float32 [H,W,3] 值域 [0,1]
        bg_color: (R,G,B) 值域 [0,1]
        alpha_low: 黑点（低于此值的 alpha 映射为 0）
        alpha_high: 白点（高于此值的 alpha 映射为 1）

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

    alpha_safe = np.maximum(alpha, epsilon)
    foreground = bg + diff / alpha_safe[..., np.newaxis]
    foreground = np.clip(foreground, 0.0, 1.0)

    transparent = alpha < epsilon
    foreground[transparent] = 0.0

    return foreground.astype(np.float32), alpha.astype(np.float32)


class RuiUnmult:
    """半透明抠图（Unmult）：指定背景色，纯数学去底，输出 RGBA 图像 + Alpha 蒙版。"""

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
                # 用中文参数名 + display:slider，界面上就是「黑点／白点」两条滑块，
                # 与 LayerStyle 的 BiRefNet Ultra 观感一致。
                # 中文名不能直接做函数形参，故 unmult() 统一用 **kwargs 接收。
                "黑点": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "display": "slider",
                    "tooltip": "黑点（主体保护）：低于此值的 alpha 强制归零。\n"
                               "调高可清除背景残留噪点，但过高会丢失边缘细节。"
                }),
                "白点": ("FLOAT", {
                    "default": 1.0, "min": 0.3, "max": 1.0, "step": 0.01,
                    "display": "slider",
                    "tooltip": "白点（主体保护）：高于此值的 alpha 强制归一。\n"
                               "调低可让主体更实、减少半透明损失，但过低会让边缘硬化。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba_image", "alpha")
    FUNCTION = "unmult"
    CATEGORY = "Rui-Node🐶/抠图✂️"

    def unmult(self, **kwargs):
        # 「黑点／白点」是中文参数名，无法直接写进函数签名，统一从 kwargs 取。
        # 同时兼容旧的英文名，避免早先保存的工作流失配。
        image = kwargs.get("image")
        bg_color = kwargs.get("bg_color", "#000000")
        alpha_low = kwargs.get("黑点", kwargs.get("alpha_low", 0.0))
        alpha_high = kwargs.get("白点", kwargs.get("alpha_high", 1.0))

        bg_rgb = _hex_to_rgb01(bg_color)

        B, H, W, C = image.shape
        if C == 4:
            image = image[..., :3]
        elif C == 1:
            image = image.repeat(1, 1, 1, 3)

        fg_list = []
        alpha_list = []

        for i in range(B):
            frame = image[i].cpu().numpy().astype(np.float32)
            fg, a = _unmult_frame(frame, bg_rgb,
                                  float(alpha_low), float(alpha_high))
            rgba = np.concatenate([fg, a[..., np.newaxis]], axis=-1)
            fg_list.append(torch.from_numpy(rgba))
            alpha_list.append(torch.from_numpy(a))

        rgba_out = torch.stack(fg_list)       # [B, H, W, 4]
        alpha_out = torch.stack(alpha_list)   # [B, H, W]

        return (rgba_out, alpha_out)


NODE_CLASS_MAPPINGS = {
    "RuiUnmult": RuiUnmult,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiUnmult": "半透明抠图 / Unmult Matting",
}
