# -*- coding: utf-8 -*-
"""
字体扫描与加载
==============
- 正文字体来自 Ruinode/font 目录（用户自备，ttf/otf/ttc），目录为空时回退系统常用中文字体。
- 自动配对同族粗体/斜体文件（msyh -> msyhbd，arial -> arialbd/ariali/arialbi），
  配不到粗体时渲染层用描边模拟。
- Emoji 用系统彩色字体（Windows: seguiemj.ttf），代码块用系统等宽字体（consola.ttf）。
"""
import os

from PIL import ImageFont

try:  # Raqm 可用时启用复杂文本布局（ZWJ 组合 emoji 等），不可用则静默用基础布局
    from PIL import features as _features
    _HAS_RAQM = bool(_features.check("raqm"))
except Exception:
    _HAS_RAQM = False

_HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.abspath(os.path.join(_HERE, "..", "font"))
_WIN_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
_EXTS = (".ttf", ".otf", ".ttc", ".otc")

# 常见字体文件名 -> 友好显示名
_FRIENDLY = {
    "msyh": "微软雅黑", "msyhbd": "微软雅黑 粗体", "msyhl": "微软雅黑 细体",
    "deng": "等线", "dengb": "等线 粗体", "dengl": "等线 细体",
    "simhei": "黑体", "simsun": "宋体", "simkai": "楷体", "simfang": "仿宋",
    "simyou": "幼圆", "stxihei": "华文细黑", "stzhongs": "华文中宋",
    "arial": "Arial", "arialbd": "Arial 粗体", "ariali": "Arial 斜体",
    "arialbi": "Arial 粗斜体", "ariblk": "Arial Black", "arialn": "Arial Narrow",
    "times": "Times New Roman", "consola": "Consolas",
}
# 下拉排序优先级（前缀匹配）
_PRIORITY = ["msyh", "deng", "simhei", "simsun", "simkai", "simfang",
             "simyou", "stxihei", "arial", "times"]
# 这些文件名貌似粗体后缀实为其他用途，禁止配对为粗体（simsunb 是宋体生僻字扩展）
_BOLD_BLACKLIST = {"simsunb"}

DEFAULT_FONT_LABEL = "微软雅黑 (msyh)"


def _rank(item):
    stem = os.path.splitext(os.path.basename(item[1]))[0].lower()
    for i, p in enumerate(_PRIORITY):
        if stem.startswith(p):
            return (i, stem)
    return (len(_PRIORITY), stem)


def scan_fonts():
    """扫描可选正文字体，返回 {显示名: 绝对路径}（有序）。"""
    found = {}
    if os.path.isdir(FONT_DIR):
        for fn in sorted(os.listdir(FONT_DIR)):
            if fn.lower().endswith(_EXTS):
                stem = os.path.splitext(fn)[0]
                friendly = _FRIENDLY.get(stem.lower())
                disp = f"{friendly} ({stem})" if friendly else stem
                found[disp] = os.path.join(FONT_DIR, fn)
    if not found:  # font 目录为空：回退系统常用中文字体
        for fn in ("msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"):
            p = os.path.join(_WIN_FONTS, fn)
            if os.path.isfile(p):
                stem = os.path.splitext(fn)[0]
                friendly = _FRIENDLY.get(stem, stem)
                found[f"{friendly} ({stem}) [系统]"] = p
    if not found:  # 彻底没有：内置位图字体兜底（不支持中文，仅保证节点可用）
        found["(内置默认字体，请向 Ruinode/font 放入 ttf)"] = ""
    return dict(sorted(found.items(), key=_rank))


def _dir_lower_map(d):
    try:
        return {fn.lower(): fn for fn in os.listdir(d)}
    except OSError:
        return {}


def resolve_family(regular_path):
    """
    由正文字体推导同族变体，返回
    {"regular": path, "bold": path|None, "italic": path|None, "bold_italic": path|None}
    """
    fam = {"regular": regular_path, "bold": None, "italic": None, "bold_italic": None}
    if not regular_path:
        return fam
    d = os.path.dirname(regular_path)
    stem = os.path.splitext(os.path.basename(regular_path))[0].lower()
    lm = _dir_lower_map(d)

    def find(stems):
        for s in stems:
            if s in _BOLD_BLACKLIST:
                continue
            for e in _EXTS:
                real = lm.get(s + e)
                if real:
                    return os.path.join(d, real)
        return None

    fam["bold"] = find([stem + "bd", stem + "b", stem + "-bold", stem + "_bold"])
    fam["italic"] = find([stem + "i", stem + "-italic", stem + "_italic"])
    fam["bold_italic"] = find([stem + "bi", stem + "z", stem + "-bolditalic"])
    return fam


def emoji_path():
    """彩色 Emoji 字体路径；找不到返回 None（届时 emoji 按普通字形绘制）。"""
    cands = [
        os.path.join(FONT_DIR, "seguiemj.ttf"),
        os.path.join(_WIN_FONTS, "seguiemj.ttf"),
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


def mono_path(fallback):
    """等宽代码字体路径；找不到时退回正文字体。"""
    cands = [
        os.path.join(FONT_DIR, "consola.ttf"),
        os.path.join(_WIN_FONTS, "consola.ttf"),
        os.path.join(_WIN_FONTS, "cour.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return fallback


_cache = {}


def load(path, size):
    """加载字体（带缓存）。path 为空/失败时退回 PIL 内置字体。"""
    size = max(4, int(round(size)))
    key = (path, size)
    if key in _cache:
        return _cache[key]
    font = None
    if path:
        layout = getattr(ImageFont, "Layout", None)
        engines = ([layout.RAQM] if (_HAS_RAQM and layout) else []) + \
                  ([layout.BASIC] if layout else []) + [None]
        for engine in engines:
            try:
                if engine is None:
                    font = ImageFont.truetype(path, size, index=0)
                else:
                    font = ImageFont.truetype(path, size, index=0, layout_engine=engine)
                break
            except Exception:
                font = None
    if font is None:
        try:
            font = ImageFont.load_default(size)
        except TypeError:  # 旧版 Pillow 无 size 参数
            font = ImageFont.load_default()
    _cache[key] = font
    return font
