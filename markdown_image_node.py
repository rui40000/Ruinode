# -*- coding: utf-8 -*-
"""
Markdown 转文本图片节点
=======================
输入 Markdown 文本，按阅读器级排版输出一张图片（IMAGE）。

- 尺寸：常用尺寸下拉快选，或选 custom 后用 width/height 精确指定
- 字体：下拉列表来自 Ruinode/font 目录（ttf/otf/ttc），刷新页面即可看到新放入的字体
- 各级字号（正文 + H1~H6）、字间距、行间距、单行字数上限均支持「auto」或手动数值，
  auto 会按输出尺寸自动求最合适的值（字号用二分法恰好优雅填满画布）
- 表格、Emoji、代码块、引用、任务清单等完整支持，视觉规范对标 GitHub 阅读器
"""
import numpy as np
import torch

from .mdimg import scan_fonts, MarkdownImageRenderer
from .mdimg.fonts import DEFAULT_FONT_LABEL

_SIZE_PRESETS = {
    "custom（使用下方宽高）": None,
    "1080×1440 竖版 3:4": (1080, 1440),
    "1080×1350 竖版 4:5": (1080, 1350),
    "1080×1920 手机 9:16": (1080, 1920),
    "1080×1080 方形 1:1": (1080, 1080),
    "1920×1080 横屏 16:9": (1920, 1080),
    "1280×720 横屏 720P": (1280, 720),
    "1200×630 链接封面": (1200, 630),
    "2480×3508 A4 纵向": (2480, 3508),
}

_THEMES = {"浅色 light": "light", "深色 dark": "dark", "米色 sepia": "sepia"}

_DEMO_MD = """# Markdown 转图片

支持 **粗体**、*斜体*、`行内代码`、~~删除线~~ 与 [链接](https://example.com)。

## 列表与任务 ✅

- 第一项：支持 Emoji 😀🎉
- 第二项
  - 嵌套子项
- [x] 已完成任务
- [ ] 待办任务

## 表格 📊

| 模型 | 输入价 | 输出价 |
|:-----|:------:|-------:|
| GPT 🤖 | $0.2/M | $1.25/M |
| Claude 🧠 | $10/M | $50/M |

> 引用块：优雅的细节，来自阅读器级的排版。

```python
def hello():
    print("Hello, Markdown!")
```
"""


def _auto_num(s, lo=None, hi=None):
    """'auto'/空 -> None；否则解析为 float 并夹取范围。"""
    t = str(s or "").strip().lower()
    if t in ("", "auto", "自动", "none"):
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


class MarkdownToImageNode:
    """Markdown 文本 -> 排版图片。"""

    @classmethod
    def INPUT_TYPES(cls):
        font_labels = list(scan_fonts().keys())
        default_font = DEFAULT_FONT_LABEL if DEFAULT_FONT_LABEL in font_labels \
            else font_labels[0]
        def _sz(desc):
            """auto 类参数：填 auto 走自动推算，填数值则强制指定。"""
            return {"default": "auto", "multiline": False,
                    "tooltip": desc + "\n填 auto 由节点自动推算，填数值则强制指定。"}

        return {
            "required": {
                "markdown": ("STRING", {
                    "default": _DEMO_MD, "multiline": True,
                    "tooltip": "Markdown 原文。支持标题、粗斜体、删除线、行内代码、\n"
                               "代码块、引用、列表、任务清单、表格、链接与彩色 Emoji。\n"
                               "⚠ 别用 WAS 的「Text Multiline」喂本节点——它会把 # 开头的\n"
                               "标题行当注释删掉。请用本仓库的「多行文本框(原样输出)」。"
                }),
                "size_preset": (list(_SIZE_PRESETS.keys()), {
                    "default": "1080×1440 竖版 3:4",
                    "tooltip": "输出尺寸预设。选 custom 时才使用下方的宽高，\n"
                               "选其他预设时下方宽高会被忽略。"
                }),
                "width": ("INT", {
                    "default": 1080, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "自定义宽度，仅在预设选 custom 时生效。"
                }),
                "height": ("INT", {
                    "default": 1440, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "自定义高度，仅在预设选 custom 时生效。\n"
                               "内容排不下时会自动缩小字号来适配，而不是裁切。"
                }),
                "font": (font_labels, {
                    "default": default_font,
                    "tooltip": "正文字体，列表来自 Ruinode/font 目录（ttf/otf/ttc）。\n"
                               "放入新字体后刷新页面即出现。同族粗体文件（如 msyhbd）\n"
                               "会自动配对用于渲染粗体，没有粗体文件时用描边模拟。"
                }),
                "theme": (list(_THEMES.keys()), {
                    "default": "浅色 light",
                    "tooltip": "配色主题：浅色适合分享长图，深色适合代码类内容，\n"
                               "米色接近纸质阅读观感。"
                }),
            },
            "optional": {
                "body_size": ("STRING", _sz("正文字号（像素）。")),
                "h1_size": ("STRING", _sz("一级标题字号。auto 时为正文的 2.0 倍。")),
                "h2_size": ("STRING", _sz("二级标题字号。auto 时为正文的 1.5 倍。")),
                "h3_size": ("STRING", _sz("三级标题字号。auto 时为正文的 1.25 倍。")),
                "h4_size": ("STRING", _sz("四级标题字号。auto 时与正文同大。")),
                "h5_size": ("STRING", _sz("五级标题字号。auto 时为正文的 0.875 倍。")),
                "h6_size": ("STRING", _sz("六级标题字号。auto 时为正文的 0.85 倍。")),
                "letter_spacing": ("STRING", _sz("字间距（像素），可为负让字更紧凑。")),
                "line_spacing": ("STRING", _sz(
                    "行距倍数，1.6 左右最接近阅读器观感；\n值越大越松散。")),
                "max_chars_per_line": ("STRING", _sz(
                    "单行最多排多少字，超出即换行。\n调小可让版面更窄、更易读。")),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "render"
    CATEGORY = "Rui-Node🐶/文本处理📝"

    @classmethod
    def VALIDATE_INPUTS(cls, font, size_preset):
        """字体/预设列表可能随目录或版本变化，宽松放行（运行时兜底）。"""
        return True

    def render(self, markdown, size_preset, width, height, font, theme,
               body_size="auto", h1_size="auto", h2_size="auto", h3_size="auto",
               h4_size="auto", h5_size="auto", h6_size="auto",
               letter_spacing="auto", line_spacing="auto",
               max_chars_per_line="auto"):
        # ---- 尺寸 ----
        preset = _SIZE_PRESETS.get(size_preset)
        w, h = preset if preset else (int(width), int(height))
        # ---- 字体 ----
        fmap = scan_fonts()
        font_path = fmap.get(font)
        if font_path is None:
            font_path = next(iter(fmap.values()))
            print(f"[Rui-Node] 字体 '{font}' 不在列表中，已回退 {font_path or '内置默认'}")
        # ---- 参数 ----
        sizes = {
            "body": _auto_num(body_size, 6, 300),
            "h1": _auto_num(h1_size, 6, 400), "h2": _auto_num(h2_size, 6, 400),
            "h3": _auto_num(h3_size, 6, 400), "h4": _auto_num(h4_size, 6, 400),
            "h5": _auto_num(h5_size, 6, 400), "h6": _auto_num(h6_size, 6, 400),
        }
        renderer = MarkdownImageRenderer(
            width=w, height=h, font_path=font_path,
            theme=_THEMES.get(theme, "light"),
            sizes=sizes,
            letter_spacing=_auto_num(letter_spacing, -10, 100),
            line_spacing=_auto_num(line_spacing, 0.8, 4.0),
            max_chars=_auto_num(max_chars_per_line, 1, 1000),
        )
        img = renderer.render(markdown)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # BHWC
        return (tensor,)


NODE_CLASS_MAPPINGS = {
    "MarkdownToImage": MarkdownToImageNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MarkdownToImage": "Markdown转图片 / Markdown To Image",
}
