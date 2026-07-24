# -*- coding: utf-8 -*-
"""
Markdown 排版渲染器（纯 PIL）
============================
视觉规范对标 GitHub / Typora 阅读器：
- 标题层级字号比例 2.0 / 1.5 / 1.25 / 1.0 / 0.875 / 0.85，h1、h2 带底部分隔线
- 正文行高 1.65，段距 0.9em，页边距为宽度的 5.5%
- 引用块左竖条 + 弱化文字色；代码块圆角底色 + 等宽字体 + 语言标签
- 表格圆角外框、表头底色加粗、斑马纹、列宽自适应（超宽自动压缩换行）
- 列表实心/空心/方形三级项目符号，任务清单绘制勾选框
- Emoji 用系统彩色字体渲染，粗体优先真字重、无字重文件时描边模拟

排版流程：解析 -> （自动模式）二分基准字号 -> 生成绘制指令 -> 绘制。
"""
import re

from PIL import Image, ImageDraw

from . import fonts
from .parser import parse_blocks

THEMES = {
    "light": dict(bg="#ffffff", fg="#1f2328", muted="#59636e", border="#d1d9e0",
                  code_bg="#f6f8fa", icode_bg="#eff1f3", quote_bar="#d1d9e0",
                  link="#0969da", hr="#d8dee4", head_bg="#f6f8fa", zebra="#f6f8fa",
                  accent="#0969da"),
    "dark": dict(bg="#0d1117", fg="#e6edf3", muted="#9198a1", border="#3d444d",
                 code_bg="#161b22", icode_bg="#2a313c", quote_bar="#3d444d",
                 link="#4493f8", hr="#30363d", head_bg="#161b22", zebra="#151b23",
                 accent="#4493f8"),
    "sepia": dict(bg="#f8f1e3", fg="#3d3226", muted="#7a6a55", border="#dccbb0",
                  code_bg="#efe5d0", icode_bg="#ece0c8", quote_bar="#d5c4a5",
                  link="#0f6674", hr="#dccbb0", head_bg="#efe5d0", zebra="#f2e9d8",
                  accent="#8a6a3a"),
}

HEAD_RATIO = {1: 2.0, 2: 1.5, 3: 1.25, 4: 1.0, 5: 0.875, 6: 0.85}

# Emoji 序列（含变体选择符、肤色、ZWJ 组合、旗帜）
_EMOJI_RE = re.compile(
    "(?:[\U0001F1E6-\U0001F1FF]{2})"
    "|(?:[©®‼⁉™ℹ↔-↪⌚-⏺"
    "Ⓜ▪-◾☀-➿⤴⤵⬀-⯿〰〽"
    "㊗㊙\U0001F000-\U0001FAFF]"
    "[️]?[\U0001F3FB-\U0001F3FF]?"
    "(?:‍[☀-➿\U0001F000-\U0001FAFF][️]?[\U0001F3FB-\U0001F3FF]?)*)"
)
_WORD_RE = re.compile(r"[A-Za-z0-9À-ɏ]+(?:['’\-_.][A-Za-z0-9À-ɏ]+)*")


def _clusters(text):
    """把文本切成排版簇：emoji 序列 / 拉丁词 / 空白 / 单个 CJK 或其他字符。"""
    out = []
    i, n = 0, len(text)
    while i < n:
        m = _EMOJI_RE.match(text, i)
        if m:
            out.append(("emoji", m.group(0)))
            i = m.end()
            continue
        m = _WORD_RE.match(text, i)
        if m:
            out.append(("word", m.group(0)))
            i = m.end()
            continue
        ch = text[i]
        out.append(("space" if ch in " \t" else "char", ch))
        i += 1
    return out


class MarkdownImageRenderer:
    """把 Markdown 文本渲染为 PIL Image。"""

    def __init__(self, width, height, font_path, theme="light",
                 sizes=None, letter_spacing=None, line_spacing=None,
                 max_chars=None):
        """
        sizes: {"body":px|None, "h1":..~"h6":..}，None 表示自动
        letter_spacing: None 自动(0) 或像素值
        line_spacing:   None 自动 或行高倍数（覆盖正文/列表/引用/表格）
        max_chars:      None 自动(不限) 或单行字数上限
        """
        self.W, self.H = int(width), int(height)
        self.pal = THEMES.get(theme, THEMES["light"])
        self.fam = fonts.resolve_family(font_path)
        self.mono_fam = fonts.resolve_family(fonts.mono_path(font_path))
        self.emoji_path = fonts.emoji_path()
        self.manual = {k: v for k, v in (sizes or {}).items() if v}
        self.lsp = float(letter_spacing or 0)
        self.lmult = line_spacing
        self.maxc = int(max_chars) if max_chars else 0
        self.pad = int(min(max(self.W * 0.055, 24), 120))
        self.avail_w = self.W - 2 * self.pad
        self.avail_h = self.H - 2 * self.pad
        self._mcache = {}
        self.overflow = False

    # ────────── 字体与测量 ──────────

    def _path_of(self, fkey):
        if fkey.startswith("mono"):
            fam = self.mono_fam
            key = {"mono": "regular", "mono_bold": "bold"}[fkey]
            return fam[key] or fam["regular"]
        p = self.fam.get(fkey)
        if not p:
            if fkey == "bold_italic":
                p = self.fam.get("bold") or self.fam.get("italic")
            p = p or self.fam["regular"]
        return p

    def _font(self, fkey, size):
        return fonts.load(self._path_of(fkey), size)

    def _tok_render(self, tok, base_size):
        """token -> (fkey, 绘制字号, 模拟粗体描边宽)。"""
        size = base_size * (0.875 if tok["code"] else 1.0)
        if tok["code"]:
            fkey = "mono_bold" if tok["bold"] and self.mono_fam["bold"] else "mono"
            stroke = 0 if (not tok["bold"] or self.mono_fam["bold"]) else max(1, round(size * 0.03))
            return fkey, size, stroke
        stroke = 0
        if tok["bold"] and tok["italic"]:
            fkey = "bold_italic" if self.fam["bold_italic"] else \
                   ("bold" if self.fam["bold"] else "italic" if self.fam["italic"] else "regular")
            if not (self.fam["bold_italic"] or self.fam["bold"]):
                stroke = max(1, round(size * 0.03))
        elif tok["bold"]:
            fkey = "bold" if self.fam["bold"] else "regular"
            if not self.fam["bold"]:
                stroke = max(1, round(size * 0.03))
        elif tok["italic"]:
            fkey = "italic" if self.fam["italic"] else "regular"
        else:
            fkey = "regular"
        return fkey, size, stroke

    def _width(self, cluster, fkey, size):
        key = (cluster, fkey, round(size, 1))
        w = self._mcache.get(key)
        if w is None:
            if fkey == "emoji":
                if self.emoji_path:
                    w = fonts.load(self.emoji_path, size).getlength(cluster)
                else:
                    w = self._font("regular", size).getlength(cluster)
            else:
                w = self._font(fkey, size).getlength(cluster)
            self._mcache[key] = w
        return w

    def _metrics(self, fkey, size):
        key = ("__m__", fkey, round(size, 1))
        m = self._mcache.get(key)
        if m is None:
            m = self._font(fkey, size).getmetrics()
            self._mcache[key] = m
        return m  # (ascent, descent)

    # ────────── 行内排版 ──────────

    def _prep_tokens(self, tokens):
        """图片 token 转为占位文本。"""
        out = []
        for t in tokens:
            if t.get("image"):
                t = dict(t, text="🖼 " + (t["text"] or "图片"), image=False, muted=True)
            out.append(t)
        return out

    def layout_inline(self, tokens, avail_w, size, mult):
        """
        内联 token -> 行列表。每行 (segs, line_w)，
        seg = (cluster, kind, tok, fkey, draw_size, w, stroke)。
        """
        segs_all = []
        for tok in self._prep_tokens(tokens):
            fkey, dsize, stroke = self._tok_render(tok, size)
            for kind, cl in _clusters(tok["text"]):
                if kind == "emoji":
                    fk = "emoji"
                elif fkey.startswith("mono") and not cl.isascii():
                    # 等宽字体通常无中文字形：非 ASCII 簇回退正文字体
                    fk = "bold" if (tok["bold"] and self.fam["bold"]) else "regular"
                else:
                    fk = fkey
                w = self._width(cl, fk, dsize) + self.lsp
                segs_all.append((cl, kind, tok, fk, dsize, w, stroke))

        lines, cur, cur_w, cur_n = [], [], 0.0, 0
        limit = self.maxc if self.maxc > 0 else 10 ** 9

        def flush():
            nonlocal cur, cur_w, cur_n
            while cur and cur[-1][0][1] == "space":  # 去行尾空白
                cur.pop()
            lines.append(([s for s, _ in cur], sum(s[5] for s, _ in cur)))
            cur, cur_w, cur_n = [], 0.0, 0

        for seg in segs_all:
            cl, kind, tok, fk, dsize, w, stroke = seg
            n_chars = 1 if kind == "emoji" else len(cl)
            if kind == "space" and not cur:
                continue  # 行首空白丢弃
            # 超宽长词字符级硬折
            if kind == "word" and w > avail_w and len(cl) > 1:
                for ch in cl:
                    cw = self._width(ch, fk, dsize) + self.lsp
                    if cur and (cur_w + cw > avail_w or cur_n + 1 > limit):
                        flush()
                    cur.append(((ch, "char", tok, fk, dsize, cw, stroke), None))
                    cur_w += cw
                    cur_n += 1
                continue
            if cur and (cur_w + w > avail_w or cur_n + n_chars > limit):
                flush()
                if kind == "space":
                    continue
            cur.append((seg, None))
            cur_w += w
            cur_n += n_chars
        if cur:
            flush()
        if not lines:
            lines.append(([], 0.0))
        return lines

    def _emit_lines(self, ops, lines, x0, avail_w, size, mult, color,
                    main_fkey="regular", align="left"):
        """把行列表落为绘制指令，返回消耗的总高度。"""
        asc, desc = self._metrics(main_fkey, size)
        line_h = size * mult
        y = self._y
        for segs, line_w in lines:
            if align == "center":
                x = x0 + max(0, (avail_w - line_w) / 2)
            elif align == "right":
                x = x0 + max(0, avail_w - line_w)
            else:
                x = x0
            baseline = y + asc + max(0, (line_h - (asc + desc)) / 2)
            # 行内代码背景（合并相邻 code 簇）
            run_s, run_e, run_size = None, None, size
            xx = x
            for cl, kind, tok, fk, dsize, w, stroke in segs:
                if tok["code"]:
                    if run_s is None:
                        run_s = xx
                    run_e = xx + w
                    run_size = dsize
                else:
                    if run_s is not None:
                        self._icode_bg(ops, run_s, run_e, baseline, run_size)
                        run_s = None
                xx += w
            if run_s is not None:
                self._icode_bg(ops, run_s, run_e, baseline, run_size)
            # 文字
            xx = x
            for cl, kind, tok, fk, dsize, w, stroke in segs:
                fill = self.pal["link"] if tok.get("link") else \
                    (self.pal["muted"] if tok.get("muted") else color)
                if kind == "emoji":
                    ops.append(("emoji", xx, baseline, cl, dsize))
                elif kind != "space":
                    ops.append(("text", xx, baseline, cl, fk, dsize, fill, stroke))
                if tok.get("strike") and kind != "space":
                    yy = baseline - dsize * 0.30
                    ops.append(("line", xx - self.lsp * 0.5, yy,
                                xx + w - self.lsp * 0.5, yy, fill,
                                max(1, round(dsize * 0.055))))
                xx += w
            y += line_h
        used = y - self._y
        self._y = y
        return used

    def _icode_bg(self, ops, x0, x1, baseline, size):
        h = size * 1.32
        y0 = baseline - size * 0.98
        ops.append(("rect", x0 - size * 0.18, y0, x1 + size * 0.18, y0 + h,
                    self.pal["icode_bg"], None, 0, size * 0.24, None))

    # ────────── 块布局 ──────────

    def _mult(self, kind):
        if self.lmult:
            return self.lmult
        return {"body": 1.65, "head": 1.3, "code": 1.55, "table": 1.45}[kind]

    def _gap(self, px):
        if not self._first_block:
            self._y += max(self._pending, px)
        self._pending = 0
        self._first_block = False

    def layout_doc(self, blocks, base):
        """给定基准字号，生成绘制指令与内容总高。"""
        self.sizes = {"body": self.manual.get("body", base)}
        b = self.sizes["body"]
        for lv in range(1, 7):
            self.sizes[f"h{lv}"] = self.manual.get(f"h{lv}", b * HEAD_RATIO[lv])
        self._y = 0.0
        self._pending = 0.0
        self._first_block = True
        ops = []
        for blk in blocks:
            self._layout_block(ops, blk, 0, self.avail_w, quote_depth=0)
        return ops, self._y

    def _layout_block(self, ops, blk, x, avail, quote_depth):
        b = self.sizes["body"]
        color = self.pal["muted"] if quote_depth else self.pal["fg"]
        t = blk["type"]

        if t == "heading":
            size = self.sizes[f"h{blk['level']}"]
            self._gap(b * 1.25)
            fkey = "bold" if self.fam["bold"] else "regular"
            toks = [dict(tk, bold=True) for tk in blk["inline"]]
            col = self.pal["muted"] if blk["level"] == 6 else color
            lines = self.layout_inline(toks, avail, size, self._mult("head"))
            self._emit_lines(ops, lines, x, avail, size, self._mult("head"), col, fkey)
            if blk["level"] in (1, 2):
                self._y += b * 0.35
                ops.append(("line", x, self._y, x + avail, self._y,
                            self.pal["hr"], max(1, round(b * 0.06))))
                self._y += b * 0.15
            self._pending = b * 0.55

        elif t == "paragraph":
            self._gap(b * 0.55)
            for ln in blk["lines"]:
                lines = self.layout_inline(ln, avail, b, self._mult("body"))
                self._emit_lines(ops, lines, x, avail, b, self._mult("body"), color)
            self._pending = b * 0.9

        elif t == "hr":
            self._gap(b * 1.3)
            yy = self._y + b * 0.1
            ops.append(("line", x, yy, x + avail, yy, self.pal["hr"],
                        max(2, round(b * 0.12))))
            self._y = yy + b * 0.1
            self._pending = b * 1.3

        elif t == "code":
            self._gap(b * 0.9)
            self._layout_code(ops, blk, x, avail)
            self._pending = b * 0.9

        elif t == "quote":
            self._gap(b * 0.9)
            top = self._y
            bar_w = max(3, round(b * 0.22))
            inner_x = x + bar_w + b * 0.9
            self._pending = 0
            self._first_block = True
            for child in blk["children"]:
                self._layout_block(ops, child, inner_x, avail - (inner_x - x),
                                   quote_depth + 1)
            ops.append(("rect", x, top + b * 0.1, x + bar_w, self._y - b * 0.05,
                        self.pal["quote_bar"], None, 0, bar_w / 2, None))
            self._pending = b * 0.9

        elif t == "list":
            self._gap(b * 0.55)
            self._layout_list(ops, blk, x, avail, depth=0, color=color)
            self._pending = b * 0.9

        elif t == "table":
            self._gap(b * 0.9)
            self._layout_table(ops, blk, x, avail)
            self._pending = b * 0.9

    # ── 代码块 ──
    def _layout_code(self, ops, blk, x, avail):
        b = self.sizes["body"]
        size = b * 0.875
        mult = self._mult("code")
        pad = b * 0.85
        asc, desc = self._metrics("mono", size)
        line_h = size * mult
        inner_w = avail - 2 * pad

        def ch_font(ch):
            # Consolas 等等宽字体无中文字形：非 ASCII 用正文字体
            return "mono" if ch.isascii() else "regular"

        # 逐字符折行，行内按 (字体,字符串) 分段
        vis_lines = []  # 每行: [(fkey, run_str, run_w), ...]
        for raw in blk["lines"]:
            cur, cur_w = [], 0.0  # cur: [(fkey, str, w)]
            if not raw:
                vis_lines.append([])
                continue
            for ch in raw:
                fk = ch_font(ch)
                cw = self._width(ch, fk, size)
                if cur and cur_w + cw > inner_w:
                    vis_lines.append(cur)
                    cur, cur_w = [], 0.0
                if cur and cur[-1][0] == fk:
                    cur[-1] = (fk, cur[-1][1] + ch, cur[-1][2] + cw)
                else:
                    cur.append((fk, ch, cw))
                cur_w += cw
            vis_lines.append(cur)
        if not vis_lines:
            vis_lines = [[]]
        box_h = len(vis_lines) * line_h + 2 * pad
        top = self._y
        ops.append(("rect", x, top, x + avail, top + box_h,
                    self.pal["code_bg"], self.pal["border"],
                    1, b * 0.5, None))
        if blk["lang"]:
            lw = self._width(blk["lang"], "mono", size * 0.8)
            ops.append(("text", x + avail - pad - lw, top + pad * 0.55 + size * 0.8,
                        blk["lang"], "mono", size * 0.8, self.pal["muted"], 0))
        yy = top + pad
        for runs in vis_lines:
            baseline = yy + asc + max(0, (line_h - (asc + desc)) / 2)
            xx = x + pad
            for fk, s, w in runs:
                ops.append(("text", xx, baseline, s, fk, size, self.pal["fg"], 0))
                xx += w
            yy += line_h
        self._y = top + box_h

    # ── 列表 ──
    def _layout_list(self, ops, blk, x, avail, depth, color):
        b = self.sizes["body"]
        indent = b * 1.7
        mult = self._mult("body")
        asc, desc = self._metrics("regular", b)
        num = blk["start"]
        for idx, item in enumerate(blk["items"]):
            if idx > 0 or depth > 0:
                self._y += b * 0.3
            top = self._y
            line_h = b * mult
            first_baseline = top + asc + max(0, (line_h - (asc + desc)) / 2)
            cx = x + indent * 0.42
            if item["task"] is not None:
                s = b * 0.95
                y0 = first_baseline - asc * 0.86
                r = (x + indent * 0.06, y0, x + indent * 0.06 + s, y0 + s)
                if item["task"] == "done":
                    ops.append(("rect", *r, self.pal["accent"], None, 0, s * 0.22, None))
                    ops.append(("check", r[0], r[1], s))
                else:
                    ops.append(("rect", *r, None, self.pal["border"],
                                max(1, round(b * 0.09)), s * 0.22, None))
            elif blk["ordered"]:
                label = f"{num}."
                lw = self._width(label, "regular", b)
                ops.append(("text", x + indent - lw - b * 0.45, first_baseline,
                            label, "regular", b, self.pal["muted"], 0))
            else:
                r = b * (0.14 if depth == 0 else 0.12)
                cy = first_baseline - asc * 0.32
                if depth == 0:
                    ops.append(("dot", cx, cy, r, color, True))
                elif depth == 1:
                    ops.append(("dot", cx, cy, r, color, False))
                else:
                    ops.append(("rect", cx - r * 0.9, cy - r * 0.9,
                                cx + r * 0.9, cy + r * 0.9, color, None, 0, 1, None))
            num += 1
            lines = self.layout_inline(item["inline"], avail - indent, b, mult)
            self._emit_lines(ops, lines, x + indent, avail - indent, b, mult, color)
            for child in item["children"]:
                self._y += b * 0.25
                self._layout_list(ops, child, x + indent, avail - indent,
                                  depth + 1, color)

    # ── 表格 ──
    def _layout_table(self, ops, blk, x, avail):
        b = self.sizes["body"]
        size = b * 0.95
        mult = self._mult("table")
        padx, pady = b * 0.8, b * 0.55
        ncol = len(blk["header"])
        if ncol == 0:
            return
        rows_all = [blk["header"]] + blk["rows"]

        def nat_min(cell):
            segs = []
            for tok in self._prep_tokens(cell):
                fkey, dsize, _ = self._tok_render(tok, size)
                for kind, cl in _clusters(tok["text"]):
                    fk = "emoji" if kind == "emoji" else fkey
                    segs.append(self._width(cl, fk, dsize))
            return (sum(segs), max(segs) if segs else size)

        nat = [0.0] * ncol
        mn = [size * 1.5] * ncol
        for row in rows_all:
            for j in range(ncol):
                w, m = nat_min(row[j] if j < len(row) else [])
                nat[j] = max(nat[j], w)
                mn[j] = max(mn[j], min(m, avail * 0.6))
        col_w = [nat[j] + 2 * padx for j in range(ncol)]
        total = sum(col_w)
        if total > avail:  # 压缩：先按可压缩空间比例，仍不足则按比例硬压
            shrink = [max(0.0, nat[j] - mn[j]) for j in range(ncol)]
            need = total - avail
            pool = sum(shrink)
            if pool > 0:
                take = min(need, pool)
                col_w = [col_w[j] - (shrink[j] / pool) * take for j in range(ncol)]
                need -= take
            if need > 0:
                scale = (sum(col_w) - need) / sum(col_w)
                col_w = [w * scale for w in col_w]
            total = sum(col_w)
        tab_w = total
        # 逐单元格排版
        asc, desc = self._metrics("regular", size)
        line_h = size * mult
        laid = []
        row_hs = []
        for ridx, row in enumerate(rows_all):
            cells = []
            maxlines = 1
            for j in range(ncol):
                toks = row[j] if j < len(row) else []
                if ridx == 0:  # 表头加粗（先转再排，保证测量与绘制一致）
                    toks = [dict(tk, bold=True) for tk in toks]
                inner = max(size, col_w[j] - 2 * padx)
                lines = self.layout_inline(toks, inner, size, mult)
                cells.append(lines)
                maxlines = max(maxlines, len(lines))
            laid.append(cells)
            row_hs.append(maxlines * line_h + 2 * pady)
        tab_h = sum(row_hs)
        top = self._y
        radius = b * 0.5
        # 表头底色 + 斑马纹
        ops.append(("rect", x, top, x + tab_w, top + row_hs[0],
                    self.pal["head_bg"], None, 0, radius, (True, True, False, False)))
        yy = top + row_hs[0]
        for ridx in range(1, len(rows_all)):
            if ridx % 2 == 0:
                last = ridx == len(rows_all) - 1
                ops.append(("rect", x, yy, x + tab_w, yy + row_hs[ridx],
                            self.pal["zebra"], None, 0, radius,
                            (False, False, last, last)))
            yy += row_hs[ridx]
        # 网格线
        lw = 1
        yy = top
        for ridx in range(len(rows_all) - 1):
            yy += row_hs[ridx]
            ops.append(("line", x, yy, x + tab_w, yy, self.pal["border"], lw))
        xx = x
        for j in range(ncol - 1):
            xx += col_w[j]
            ops.append(("line", xx, top, xx, top + tab_h, self.pal["border"], lw))
        ops.append(("rect", x, top, x + tab_w, top + tab_h, None,
                    self.pal["border"], lw, radius, None))
        # 文字
        for ridx, cells in enumerate(laid):
            cy = top + sum(row_hs[:ridx]) + pady
            for j in range(ncol):
                cx = x + sum(col_w[:j]) + padx
                inner = max(size, col_w[j] - 2 * padx)
                align = blk["align"][j] if j < len(blk["align"]) else "left"
                self._y = cy
                self._emit_lines(ops, cells[j], cx, inner, size, mult,
                                 self.pal["fg"], "regular", align)
        self._y = top + tab_h

    # ────────── 自动字号与总渲染 ──────────

    def _auto_base(self, blocks):
        if "body" in self.manual:
            return self.manual["body"]
        lo, hi = 10, int(min(max(self.W // 22, 18), 60))
        best = lo
        fit_any = False
        while lo <= hi:
            mid = (lo + hi) // 2
            _, h = self.layout_doc(blocks, mid)
            if h <= self.avail_h:
                best, fit_any = mid, True
                lo = mid + 1
            else:
                hi = mid - 1
        if not fit_any:
            best = 10
        return best

    def render(self, md_text):
        blocks = parse_blocks(md_text or "")
        if not blocks:
            blocks = [{"type": "paragraph",
                       "lines": [[dict(text="（空文档）", bold=False, italic=False,
                                       code=False, strike=False, link=None,
                                       image=False)]]}]
        base = self._auto_base(blocks)
        ops, content_h = self.layout_doc(blocks, base)
        self.overflow = content_h > self.avail_h + 1
        if self.overflow:
            print(f"[Rui-Node] Markdown 内容高度 {int(content_h)}px 超出画布可用高度 "
                  f"{self.avail_h}px，超出部分将被裁剪（可增大高度或调小字号）")
        img = Image.new("RGB", (self.W, self.H), self.pal["bg"])
        draw = ImageDraw.Draw(img, "RGBA")
        ox, oy = self.pad, self.pad
        for op in ops:
            kind = op[0]
            if kind == "rect":
                _, x0, y0, x1, y1, fill, outline, w, radius, corners = op
                if x1 - x0 < 1 or y1 - y0 < 1:
                    continue
                kw = dict(radius=max(0, radius), fill=fill, outline=outline,
                          width=max(1, int(w)) if outline else 0)
                if corners is not None:
                    kw["corners"] = corners
                try:
                    draw.rounded_rectangle((x0 + ox, y0 + oy, x1 + ox, y1 + oy), **kw)
                except TypeError:  # 旧版 Pillow 无 corners
                    kw.pop("corners", None)
                    draw.rounded_rectangle((x0 + ox, y0 + oy, x1 + ox, y1 + oy), **kw)
            elif kind == "line":
                _, x0, y0, x1, y1, fill, w = op
                draw.line((x0 + ox, y0 + oy, x1 + ox, y1 + oy), fill=fill,
                          width=max(1, int(round(w))))
            elif kind == "text":
                _, x, y, s, fk, size, fill, stroke = op
                f = self._font(fk, size)
                if stroke:
                    draw.text((x + ox, y + oy), s, font=f, fill=fill, anchor="ls",
                              stroke_width=int(stroke), stroke_fill=fill)
                else:
                    draw.text((x + ox, y + oy), s, font=f, fill=fill, anchor="ls")
            elif kind == "emoji":
                _, x, y, s, size = op
                self._draw_emoji(draw, x + ox, y + oy, s, size)
            elif kind == "dot":
                _, cx, cy, r, fill, solid = op
                bb = (cx + ox - r, cy + oy - r, cx + ox + r, cy + oy + r)
                if solid:
                    draw.ellipse(bb, fill=fill)
                else:
                    draw.ellipse(bb, outline=fill, width=max(1, int(r * 0.45)))
            elif kind == "check":
                _, x0, y0, s = op
                x0, y0 = x0 + ox, y0 + oy
                w = max(2, round(s * 0.14))
                pts = [(x0 + s * 0.24, y0 + s * 0.52), (x0 + s * 0.43, y0 + s * 0.72),
                       (x0 + s * 0.78, y0 + s * 0.30)]
                draw.line(pts, fill="#ffffff", width=w, joint="curve")
        return img

    def _draw_emoji(self, draw, x, y, s, size):
        if self.emoji_path:
            try:
                f = fonts.load(self.emoji_path, size)
                draw.text((x, y), s, font=f, embedded_color=True, anchor="ls")
                return
            except Exception:
                pass
        draw.text((x, y), s, font=self._font("regular", size),
                  fill=self.pal["fg"], anchor="ls")
