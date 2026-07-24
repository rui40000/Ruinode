# -*- coding: utf-8 -*-
"""
Markdown 解析器（为图片渲染定制的轻量实现，零第三方依赖）
=========================================================
块级：标题 / 段落 / 围栏代码块 / 引用（可嵌套）/ 有序·无序·任务列表（可嵌套）/
      表格（含对齐）/ 分割线
内联：`code`、**粗**、*斜*、***粗斜***、~~删除~~、[链接](url)、![图片](url)

约定：段落内的单个换行按「硬换行」处理（贴长文的用户通常期望所见即所得），
空行分段。

块结构（dict）：
  {"type":"heading","level":1-6,"inline":[...]}
  {"type":"paragraph","lines":[[inline...],...]}
  {"type":"code","lang":str,"lines":[str,...]}
  {"type":"quote","children":[block,...]}
  {"type":"list","ordered":bool,"start":int,
   "items":[{"inline":[...],"task":None|"todo"|"done","children":[block,...]},...]}
  {"type":"table","align":["left|center|right",...],
   "header":[[inline...],...],"rows":[[[inline...],...],...]}
  {"type":"hr"}
内联 token（dict）：
  {"text":str,"bold":bool,"italic":bool,"code":bool,"strike":bool,
   "link":str|None,"image":bool}
"""
import re

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_HR_RE = re.compile(r"^ {0,3}(-{3,}|\*{3,}|_{3,})\s*$")
_FENCE_RE = re.compile(r"^ {0,3}(```+|~~~+)\s*([^\s`]*)\s*$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])\s+(.*)$")
_TASK_RE = re.compile(r"^\[([ xX])\]\s+(.*)$")
_QUOTE_RE = re.compile(r"^ {0,3}>\s?(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


# ────────────────────────── 内联解析 ──────────────────────────

_INLINE_RE = re.compile(
    r"(?P<code>`+)(?P<code_t>.+?)(?P=code)"
    r"|(?P<bi>\*\*\*|___)(?P<bi_t>[^*_]+?)(?P=bi)"
    r"|(?P<b>\*\*|__)(?P<b_t>.+?)(?P=b)"
    r"|(?P<i>\*)(?P<i_t>[^*]+?)\*"
    r"|(?<![0-9A-Za-z_])_(?P<iu_t>[^_]+?)_(?![0-9A-Za-z_])"
    r"|(?P<s>~~)(?P<s_t>.+?)~~"
    r"|(?P<img>!)?\[(?P<lk_t>[^\]]*)\]\((?P<lk_u>[^)\s]*)[^)]*\)"
)


def _tok(text, base, **over):
    t = dict(base)
    t.update(over)
    t["text"] = text
    return t


_BASE_STYLE = {"text": "", "bold": False, "italic": False, "code": False,
               "strike": False, "link": None, "image": False}


def parse_inline(text, base=None):
    """把一行文本解析为内联 token 列表。"""
    base = dict(base or _BASE_STYLE)
    out = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            out.append(_tok(text[pos:m.start()], base))
        g = m.groupdict()
        if g["code"] is not None:
            out.append(_tok(g["code_t"], base, code=True))
        elif g["bi"] is not None:
            out.extend(parse_inline(g["bi_t"], {**base, "bold": True, "italic": True}))
        elif g["b"] is not None:
            out.extend(parse_inline(g["b_t"], {**base, "bold": True}))
        elif g["i_t"] is not None:
            out.extend(parse_inline(g["i_t"], {**base, "italic": True}))
        elif g["iu_t"] is not None:
            out.extend(parse_inline(g["iu_t"], {**base, "italic": True}))
        elif g["s"] is not None:
            out.extend(parse_inline(g["s_t"], {**base, "strike": True}))
        elif g["lk_t"] is not None:
            if g["img"]:
                out.append(_tok(g["lk_t"] or "图片", base, image=True, link=g["lk_u"]))
            else:
                out.extend(parse_inline(g["lk_t"], {**base, "link": g["lk_u"] or "#"}))
        pos = m.end()
    if pos < len(text):
        out.append(_tok(text[pos:], base))
    return [t for t in out if t["text"] != "" or t["image"]]


# ────────────────────────── 块级解析 ──────────────────────────

def _split_cells(line):
    """拆表格行的单元格（支持 \\| 转义）。"""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells, cur, esc = [], [], False
    for ch in s:
        if esc:
            cur.append(ch)
            esc = False
        elif ch == "\\":
            esc = True
        elif ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def _parse_align(sep_line):
    aligns = []
    for c in _split_cells(sep_line):
        c = c.strip()
        left, right = c.startswith(":"), c.endswith(":")
        aligns.append("center" if left and right else
                      "right" if right else "left")
    return aligns


def _indent_width(s):
    w = 0
    for ch in s:
        if ch == " ":
            w += 1
        elif ch == "\t":
            w += 4
        else:
            break
    return w


def _is_table_start(lines, i):
    """当前行 + 下一行是否构成表格开头（单列表格如 "| a |"+"|---|" 也成立；
    纯 "---" 留给分割线）。表格分支与段落中断判定必须共用本函数，避免解析死循环。"""
    if "|" not in lines[i] or i + 1 >= len(lines):
        return False
    nxt = lines[i + 1]
    if not _TABLE_SEP_RE.match(nxt):
        return False
    return "|" in nxt or len(_split_cells(nxt)) >= 2


def _parse_list(lines, i):
    """从第 i 行开始收集列表，返回 (块列表, next_i)。
    同级项的 marker 在有序/无序间切换时拆分为独立列表块。"""
    entries = []  # (indent, marker, text)
    while i < len(lines):
        m = _LIST_RE.match(lines[i])
        if m:
            entries.append((_indent_width(m.group(1)), m.group(2), m.group(3)))
            i += 1
        elif lines[i].strip() == "":
            # 空行后若仍是列表则继续（宽松处理）
            if i + 1 < len(lines) and _LIST_RE.match(lines[i + 1]):
                i += 1
            else:
                break
        else:
            break

    def build_seq(seq):
        """同层条目 -> 列表块序列（marker 类型变化即切块）。"""
        base = seq[0][0]
        out, items, cur_ordered, cur_start = [], [], None, 1

        def close():
            nonlocal items
            if items:
                out.append({"type": "list", "ordered": cur_ordered,
                            "start": cur_start, "items": items})
                items = []

        j = 0
        while j < len(seq):
            ind, marker, text = seq[j]
            ordered = marker[0].isdigit()
            k = j + 1
            sub = []
            while k < len(seq) and seq[k][0] > base + 1:
                sub.append(seq[k])
                k += 1
            if ordered != cur_ordered:
                close()
                cur_ordered = ordered
                if ordered:
                    try:
                        cur_start = int(re.match(r"\d+", marker).group(0))
                    except Exception:
                        cur_start = 1
            task = None
            tm = _TASK_RE.match(text)
            if tm and not ordered:
                task = "done" if tm.group(1).lower() == "x" else "todo"
                text = tm.group(2)
            items.append({"inline": parse_inline(text), "task": task,
                          "children": build_seq(sub) if sub else []})
            j = k
        close()
        return out

    return build_seq(entries), i


def parse_blocks(text):
    """Markdown 全文 -> 块列表。"""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            i += 1
            continue

        # 围栏代码块
        fm = _FENCE_RE.match(line)
        if fm:
            fence, lang = fm.group(1)[0] * 3, fm.group(2)
            body = []
            i += 1
            while i < n and not _FENCE_RE.match(lines[i]):
                body.append(lines[i].replace("\t", "    "))
                i += 1
            i += 1  # 跳过闭合围栏
            blocks.append({"type": "code", "lang": lang, "lines": body})
            continue

        # 标题
        hm = _HEADING_RE.match(line)
        if hm:
            blocks.append({"type": "heading", "level": len(hm.group(1)),
                           "inline": parse_inline(hm.group(2))})
            i += 1
            continue

        # 分割线
        if _HR_RE.match(line):
            blocks.append({"type": "hr"})
            i += 1
            continue

        # 引用
        qm = _QUOTE_RE.match(line)
        if qm:
            inner = []
            while i < n:
                q = _QUOTE_RE.match(lines[i])
                if q:
                    inner.append(q.group(1))
                    i += 1
                elif lines[i].strip() != "" and inner and not _LIST_RE.match(lines[i]):
                    inner.append(lines[i].strip())  # 懒接续
                    i += 1
                else:
                    break
            blocks.append({"type": "quote", "children": parse_blocks("\n".join(inner))})
            continue

        # 表格：当前行含 | 且下一行是分隔行
        if _is_table_start(lines, i):
            header = [parse_inline(c) for c in _split_cells(line)]
            align = _parse_align(lines[i + 1])
            ncol = len(header)
            align = (align + ["left"] * ncol)[:ncol]
            rows = []
            i += 2
            while i < n and "|" in lines[i] and lines[i].strip() != "":
                cells = [parse_inline(c) for c in _split_cells(lines[i])]
                cells = (cells + [[] for _ in range(ncol)])[:ncol]
                rows.append(cells)
                i += 1
            blocks.append({"type": "table", "align": align,
                           "header": header, "rows": rows})
            continue

        # 列表（marker 类型切换会拆出多个块）
        if _LIST_RE.match(line):
            blks, i = _parse_list(lines, i)
            blocks.extend(blks)
            continue

        # 段落：收集到空行或下一个特殊块，单换行=硬换行
        para = []
        while i < n:
            cur = lines[i]
            if (cur.strip() == "" or _HEADING_RE.match(cur) or _HR_RE.match(cur)
                    or _FENCE_RE.match(cur) or _QUOTE_RE.match(cur)
                    or _LIST_RE.match(cur) or _is_table_start(lines, i)):
                break
            para.append(parse_inline(cur.strip()))
            i += 1
        if para:
            blocks.append({"type": "paragraph", "lines": para})
        else:
            i += 1  # 兜底：任何未被消费的行强制前进，杜绝解析死循环
    return blocks
