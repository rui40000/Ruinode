# -*- coding: utf-8 -*-
"""
满屏文字水印节点
================
给输入图像铺满一层平铺的文字水印，常用于版权标注 / 防盗图。

可调参数：
1. text            —— 水印文案（支持多行，用换行分隔）
2. font            —— 字体样式（下拉，来自 Ruinode/font 目录，同 Markdown 节点）
3. font_size       —— 文字大小（像素）
4. angle           —— 水印整体旋转角度（度，-180~180）
5. density         —— 水印密度（综合控制行间距与同行水印之间的间距，值越大越密）
6. letter_spacing  —— 字间距（单条文案内相邻字符的额外间距，像素，可为负）
7. opacity         —— 水印透明程度（0=完全透明，100=完全不透明）
8. color           —— 水印文字颜色（#RRGGBB / #RGB / "r,g,b" / 常见英文色名）

实现要点：在一张比原图对角线还大的透明画布上按交错网格平铺文字，
整体旋转后从中心裁出与原图等大的区域，再 alpha 合成回原图，
这样任意旋转角度下都能保证四角也被水印覆盖，不留空白。
"""
import math

import numpy as np
import torch
from PIL import Image, ImageDraw

from .mdimg.fonts import scan_fonts, load as load_font, DEFAULT_FONT_LABEL


# 常见英文色名兜底
_COLOR_NAMES = {
    "white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0),
    "green": (0, 128, 0), "blue": (0, 0, 255), "gray": (128, 128, 128),
    "grey": (128, 128, 128), "yellow": (255, 255, 0), "orange": (255, 165, 0),
    "cyan": (0, 255, 255), "magenta": (255, 0, 255), "silver": (192, 192, 192),
}


def _parse_color(s):
    """把颜色字符串解析为 (r, g, b)；无法识别时回退灰色。"""
    s = str(s or "").strip()
    if not s:
        return (128, 128, 128)
    if s.startswith("#"):
        h = s[1:].strip()
        if len(h) == 3:                       # #RGB -> #RRGGBB
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            try:
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            except ValueError:
                pass
    if "," in s:                              # "r,g,b"
        parts = [p.strip() for p in s.split(",")]
        try:
            vals = [max(0, min(255, int(float(p)))) for p in parts[:3]]
            if len(vals) == 3:
                return tuple(vals)
        except ValueError:
            pass
    return _COLOR_NAMES.get(s.lower(), (128, 128, 128))


def _line_width(font, text, spacing):
    """含字间距时单行文本的像素宽度。"""
    if not text:
        return 0.0
    w = 0.0
    for ch in text:
        w += font.getlength(ch) + spacing
    return max(0.0, w - spacing)              # 末字符不加尾随间距


def _draw_spaced_line(draw, xy, text, font, fill, spacing):
    """逐字符绘制一行文本，字与字之间插入 spacing 像素间距。"""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += font.getlength(ch) + spacing


class ImageWatermarkNode:
    """满屏文字水印。"""

    @classmethod
    def INPUT_TYPES(cls):
        font_labels = list(scan_fonts().keys())
        default_font = DEFAULT_FONT_LABEL if DEFAULT_FONT_LABEL in font_labels \
            else (font_labels[0] if font_labels else "")
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "要打水印的图像，支持批量，每张都会独立处理。"
                }),
                "text": ("STRING", {
                    "default": "仅供参考 / SAMPLE",
                    "multiline": True,
                    "tooltip": "水印文案，支持多行（直接在框里换行）。\n"
                               "多行会作为一个整体单元平铺，行内间距按字号的 25% 自动计算。"
                }),
                "font": (font_labels, {
                    "default": default_font,
                    "tooltip": "字体，列表来自 Ruinode/font 目录，与 Markdown 转图片节点共用。\n"
                               "放入新的 ttf/otf/ttc 后刷新页面即可出现。\n"
                               "列表里没有的字体会自动回退，不会报错。"
                }),
                "font_size": ("INT", {
                    "default": 48, "min": 8, "max": 500, "step": 1,
                    "tooltip": "文字大小（像素）。它同时是密度的基准——\n"
                               "改字号时水印疏密会等比缩放，视觉观感保持稳定。"
                }),
                "angle": ("FLOAT", {
                    "default": 30.0, "min": -180.0, "max": 180.0, "step": 1.0,
                    "tooltip": "水印整体旋转角度（度）。\n"
                               "实现上先在超过对角线的画布上平铺、整体旋转后再从中心裁切，\n"
                               "所以任何角度下四角都不会露白。"
                }),
                "density": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05,
                    "tooltip": "水印密度，越大越密。控制的是水印之间的空隙：\n"
                               "空隙 = 字号 × 2 ÷ 密度。\n"
                               "0.4 左右 = 稀疏点缀；1.0 = 默认；\n"
                               "2.5 以上 = 密集防盗（建议同时把透明度降到 20~30）"
                }),
                "letter_spacing": ("INT", {
                    "default": 0, "min": -20, "max": 200, "step": 1,
                    "tooltip": "字间距：单条文案内相邻字符的额外间距（像素）。\n"
                               "逐字符绘制实现，可为负值让字挤在一起。"
                }),
                "opacity": ("FLOAT", {
                    "default": 35.0, "min": 0.0, "max": 100.0, "step": 1.0,
                    "tooltip": "水印透明程度：0 = 完全透明（原样返回，不做任何计算），\n"
                               "100 = 完全不透明。\n"
                               "防盗建议 25~40：看得清但不影响读图。"
                }),
                "color": ("STRING", {
                    "default": "#FFFFFF", "multiline": False,
                    "tooltip": "水印颜色，四种写法都支持：\n"
                               "#FFFFFF ／ #FFF ／ 255,255,255 ／ white\n"
                               "认不出的写法会回退成灰色，不会报错。\n"
                               "深色图用白色，浅色图用黑或深灰，彩色图用黄色最醒目。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_watermark"
    CATEGORY = "Rui-Node🐶/图像调节🎨"

    @classmethod
    def VALIDATE_INPUTS(cls, font, **kwargs):
        """字体列表会随 font 目录变化，宽松放行，运行时兜底回退。"""
        return True

    # ---- 生成单张图的水印层并合成 ----
    def _watermark_one(self, pil_img, text, font, font_size, angle,
                       density, letter_spacing, opacity, color):
        w, h = pil_img.size
        rgb = _parse_color(color)
        alpha = int(round(max(0.0, min(100.0, opacity)) / 100.0 * 255))
        if alpha <= 0 or not text.strip():
            return pil_img.convert("RGB")            # 全透明或空文案：原样返回
        fill = (rgb[0], rgb[1], rgb[2], alpha)

        lines = text.split("\n")
        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        inner_gap = max(1, int(line_h * 0.25))       # 文案内部多行的行距
        tile_w = max((_line_width(font, ln, letter_spacing) for ln in lines),
                     default=0.0)
        tile_h = line_h * len(lines) + inner_gap * (len(lines) - 1)
        tile_w = max(tile_w, 1.0)
        tile_h = max(tile_h, 1.0)

        # density 越大间距越小：以字号为基准的空隙除以密度
        gap = max(2.0, font_size * 2.0 / max(0.1, density))
        step_x = tile_w + gap
        step_y = tile_h + gap

        # 画布要盖住旋转后的对角线，四周再留一圈平铺余量
        diag = math.ceil(math.sqrt(w * w + h * h))
        margin = int(max(step_x, step_y) + max(tile_w, tile_h)) + 10
        cw = diag + 2 * margin
        ch = diag + 2 * margin

        layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        # 交错平铺（隔行水平偏移半步），视觉更自然
        row = 0
        y = -step_y
        while y < ch + step_y:
            x_off = (step_x / 2.0) if (row % 2) else 0.0
            x = -step_x + x_off
            while x < cw + step_x:
                ty = y
                for ln in lines:
                    if ln:
                        _draw_spaced_line(draw, (x, ty), ln, font, fill,
                                          letter_spacing)
                    ty += line_h + inner_gap
                x += step_x
            y += step_y
            row += 1

        # 整体旋转后从中心裁出与原图等大的区域
        rotated = layer.rotate(angle, resample=Image.BICUBIC, expand=False)
        left = (cw - w) // 2
        top = (ch - h) // 2
        crop = rotated.crop((left, top, left + w, top + h))

        base = pil_img.convert("RGBA")
        out = Image.alpha_composite(base, crop)
        return out.convert("RGB")

    def apply_watermark(self, image, text, font, font_size, angle,
                        density, letter_spacing, opacity, color):
        # 加载字体（沿用 Markdown 节点的扫描/回退逻辑）
        fmap = scan_fonts()
        font_path = fmap.get(font)
        if font_path is None:
            font_path = next(iter(fmap.values()), "")
            print(f"[Rui-Node] 水印字体 '{font}' 不在列表中，已回退 "
                  f"{font_path or '内置默认'}")
        pil_font = load_font(font_path, font_size)

        results = []
        batch = image.shape[0]
        for i in range(batch):
            arr = np.clip(image[i].cpu().numpy(), 0.0, 1.0)
            pil_img = Image.fromarray((arr * 255).astype(np.uint8), "RGB")
            out = self._watermark_one(
                pil_img, text, pil_font, int(font_size), float(angle),
                float(density), int(letter_spacing), float(opacity), color)
            out_np = np.asarray(out, dtype=np.float32) / 255.0
            results.append(torch.from_numpy(out_np))

        return (torch.stack(results),)


NODE_CLASS_MAPPINGS = {
    "ImageWatermark": ImageWatermarkNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageWatermark": "满屏文字水印 / Full-Screen Text Watermark",
}
