# -*- coding: utf-8 -*-
"""
八方向雪碧图序列拆分节点（Ruinode）
===================================
用于「8 方向行走动画」制作管线：把每帧排布着 8 个朝向的雪碧图序列，
拆成 8 条各自独立、可直接成片的动画序列。

典型管线：
    角色图 → (GPTimage2) 八方向静态图 → (Seedance 首尾帧) 循环行走视频
    → VHS「Load Video」转序列帧（可选帧率）
    → 【本节点】拆分 + 方向编号 + 分组
    → 8×SaveImage 出序列帧，8×VHS「Video Combine」出透明 webm

为什么用固定网格而不是连通区域拆分（本仓库的 RuiSpriteSplitterRGBA）：
- 连通区域按包围盒排序，角色走动时位置会浮动，一旦跨过排序的行界，
  方向对应关系就错乱 —— 几十上百帧里错一帧，整条动画就废了；
- 连通区域按各自 bbox 裁剪，每个 sprite 尺寸不同，无法直接合成视频。
固定网格没有这两个问题：格子位置恒定，方向对应天然稳定，尺寸也一致。
（连通区域拆分依然更适合单张静态合图，两者各有用武之地。）

白底转透明的关键点：角色身上常有白色衣物，按亮度阈值一刀切会把白衬衫
一起掏空。这里改为**从画面边缘做连通性判断**：只有与边缘相连的白色才算背景，
被角色包围的白色（衣服、高光）一律保留。
"""
import numpy as np
import torch

try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except Exception:                                   # 理论上 ComfyUI 环境都有
    _HAS_SCIPY = False

MAX_DIRS = 8

_BG_MODES = {
    # 键名直接把取舍写清楚：颜色阈值法快但会啃掉白色衣物，
    # 接模型抠图才是干净的做法（实测对比见 README）
    "已带透明通道（推荐·上游接抠图节点）": "keep",
    "白底转透明（快，但白色衣物会被啃）": "white",
    "不处理（输出不透明）": "none",
}
# 早先版本保存的工作流里的旧选项名，不进下拉、只做解析兼容
_BG_ALIAS = {
    "白底转透明（推荐）": "white",
    "已带透明通道": "keep",
}

# 与 3×3 中间留空的常见排布对应：行 1 面向观众、行 3 背对观众
_DEFAULT_NAMES = "SW,S,SE,W,E,NW,N,NE"


def _white_to_alpha(rgb, threshold, softness):
    """
    白底 → alpha。rgb: (H,W,3) float[0,1]，返回 (H,W) float[0,1]。

    先按「离白色多远」算出软 alpha 保住边缘抗锯齿，再用连通性把
    与画面边缘相连的白色判为背景 —— 只有这一部分才真正抹成全透明。
    这样角色内部的白衬衫、白高光不会被误伤。
    """
    dist = 1.0 - rgb.min(axis=2)                     # 纯白=0，越大越不白
    cut = max(1e-4, 1.0 - float(threshold))
    soft = np.clip(dist / (cut * max(1e-3, softness)), 0.0, 1.0)

    near_white = dist <= cut
    if _HAS_SCIPY and near_white.any():
        lab, n = ndi.label(near_white)
        if n > 0:
            border = np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
            ids = np.unique(border)
            ids = ids[ids != 0]
            if ids.size:
                bg = np.isin(lab, ids)
                soft = np.where(bg, 0.0, soft)
    else:
        soft = np.where(near_white, 0.0, soft)
    return soft.astype(np.float32)


def _drop_fragments(alpha, ratio):
    """
    清掉远小于主体的连通碎片。

    网格切分难免会把相邻格子探过来的部件（手杖尖、飘起的衣角）切进本格，
    这些碎片不仅难看，还会把 auto_crop 的包围盒撑大。
    按「面积不足主体 ratio 倍」判定为碎片，这样与身体相连的道具不会被误删。
    """
    if ratio <= 0 or not _HAS_SCIPY:
        return alpha
    solid = alpha > 0.1
    if not solid.any():
        return alpha
    lab, n = ndi.label(solid)
    if n <= 1:
        return alpha
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    keep = areas >= areas.max() * float(ratio)
    keep[0] = False
    return np.where(keep[lab], alpha, 0.0).astype(np.float32)


def _shrink_alpha(alpha, shrink):
    """
    把边缘那圈「几乎全是背景」的半透明像素收掉。

    白底图上，角色边缘的抗锯齿像素本就是前景与白色的混合，而从单张图
    反推 alpha 是欠定问题（一个方程多个未知数），浅色边缘的 alpha 往往
    被估低，于是残留一圈混了白的半透明像素——合成到深色背景就是白边。
    这里把 alpha 低于 shrink 的判为背景，其余重新拉伸到满值，
    相当于平滑地向内收一点边，比形态学腐蚀不容易产生锯齿。
    """
    if shrink <= 0:
        return alpha
    s = min(0.95, float(shrink))
    return np.clip((alpha - s) / (1.0 - s), 0.0, 1.0).astype(np.float32)


def _decontaminate(rgb, alpha, strength, bg=1.0):
    """
    颜色反溢出：把混进边缘像素的背景色剥掉。

    观察色 = 前景色 × a + 背景色 × (1-a)，于是
        前景色 = (观察色 - 背景色 × (1-a)) / a
    不做这一步，边缘像素的 RGB 里就一直掺着白，贴到深色背景上必然发白。
    a 极小时该式会放大噪声，故对分母设下限。
    """
    if strength <= 0:
        return rgb
    a = alpha[..., None]
    fg = (rgb - bg * (1.0 - a)) / np.maximum(a, 0.08)
    fg = np.clip(fg, 0.0, 1.0)
    s = float(min(1.0, strength))
    return (rgb * (1.0 - s) + fg * s).astype(np.float32)


def _assign_blocks(alpha_full, ybnd, xbnd, cols, frag_ratio):
    """
    全图连通标记 + 按质心把每块归属到格子。

    这样格子只负责回答「这是哪个方向」，角色的实际范围由它自己的连通块决定，
    因此**角色超出格子边界也不会被切**（原本严格等分会把探出去的脚、
    飘起的斗篷直接截断）。归属按质心判定，每块只属于一个格子，
    相邻角色不会被重复计入。

    返回 (lab, {cell_id: [块标签...]})；scipy 不可用时返回 (None, None)。
    """
    if not _HAS_SCIPY:
        return None, None
    solid = alpha_full > 0.1
    if not solid.any():
        return None, {}
    lab, n = ndi.label(solid)
    if n == 0:
        return lab, {}
    areas = np.bincount(lab.ravel())
    areas[0] = 0
    big = areas.max()
    cents = ndi.center_of_mass(solid, lab, np.arange(1, n + 1))
    rows = len(ybnd) - 1
    out = {}
    for i in range(1, n + 1):
        if frag_ratio > 0 and areas[i] < big * float(frag_ratio):
            continue                                   # 碎片，丢弃
        cy, cx = cents[i - 1]
        r = min(rows - 1, max(0, int(np.searchsorted(ybnd, cy, "right") - 1)))
        c = min(cols - 1, max(0, int(np.searchsorted(xbnd, cx, "right") - 1)))
        out.setdefault(r * cols + c, []).append(i)
    return lab, out


def _bbox(alpha, thr=0.02):
    """内容包围盒 (y0,y1,x0,x1)，无内容时返回 None。"""
    m = alpha > thr
    if not m.any():
        return None
    ys = np.where(m.any(axis=1))[0]
    xs = np.where(m.any(axis=0))[0]
    return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1


def _parse_cells(text, total):
    out = set()
    for tok in str(text or "").replace("，", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if 0 <= v < total:
            out.add(v)
    return out


class RuiEightDirSplit:
    """八方向雪碧图序列 → 8 条独立动画序列。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "视频转出的序列帧，每帧是一张排布着多个朝向的雪碧图。"
                }),
                "grid_cols": ("INT", {
                    "default": 3, "min": 1, "max": 8, "step": 1,
                    "tooltip": "雪碧图的列数。"
                }),
                "grid_rows": ("INT", {
                    "default": 3, "min": 1, "max": 8, "step": 1,
                    "tooltip": "雪碧图的行数。"
                }),
                "empty_cells": ("STRING", {
                    "default": "4", "multiline": False,
                    "tooltip": "空格子的序号（行优先、从 0 开始，逗号分隔）。\n"
                               "3×3 布局中间留空即填 4。留空表示没有空格。"
                }),
                "direction_names": ("STRING", {
                    "default": _DEFAULT_NAMES, "multiline": False,
                    "tooltip": "按「跳过空格后的先后顺序」给每个方向命名，逗号分隔。\n"
                               "默认对应 3×3 中间留空、行 1 面向观众的排布：\n"
                               "  SW  S  SE\n"
                               "  W      E\n"
                               "  NW  N  NE\n"
                               "名字只用于 info 与你自己辨认，不影响画面内容。"
                }),
                "bg_mode": (list(_BG_MODES.keys()), {
                    "default": "已带透明通道（推荐·上游接抠图节点）",
                    "tooltip": "webm 要保留透明就必须先有 alpha。三种来源：\n\n"
                               "【已带透明通道】把上游抠图节点（Lucida / FeyNobg）的\n"
                               "  遮罩接到 masks 输入。**强烈推荐**：模型是按语义判断的，\n"
                               "  白衣服不会被误判成背景。实测同等白边水平下，\n"
                               "  主体被啃掉的面积比阈值法少 35%。\n"
                               "  没接 masks 时会自动退回颜色阈值法，不会报错。\n\n"
                               "【白底转透明】纯颜色阈值 + 边缘连通性判断，不需要模型、\n"
                               "  很快。但它靠「离白色多远」估 alpha，角色身上的白色衣物\n"
                               "  天生 alpha 偏低，收边时会被啃出破洞 —— 素材里有白衣服、\n"
                               "  白高光时别用这个。\n\n"
                               "【不处理】输出不透明。仍会按角色范围裁剪、不切断。"
                }),
                "bg_threshold": ("FLOAT", {
                    "default": 0.92, "min": 0.5, "max": 1.0, "step": 0.005,
                    "tooltip": "白底判定阈值：像素三通道最小值高于它才算「接近白」。\n"
                               "背景没扣干净就调低，角色边缘被啃掉就调高。"
                }),
                "edge_shrink": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 0.95, "step": 0.01,
                    "tooltip": "【治白边的主力参数】把边缘那圈「几乎全是背景」的\n"
                               "半透明像素收掉，相当于平滑地向内收一点边。\n"
                               "白底素材的边缘像素本就掺了白，不收掉贴到深色背景就发白。\n\n"
                               "配模型 alpha（bg_mode=已带透明通道）：0.15~0.25 就够。\n"
                               "配颜色阈值法：要 0.35 以上才压得住白边，代价是白色衣物\n"
                               "  会被啃出破洞 —— 这也是推荐用模型 alpha 的原因。\n"
                               "0 = 完全不收（白边明显）。"
                }),
                "decontaminate": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "【治白边的第二道】颜色反溢出：把混进边缘像素的白色剥掉。\n"
                               "原理是按 观察色 = 前景×a + 白×(1-a) 反解出真正的前景色。\n"
                               "1.0 = 完全反解（推荐），0 = 不处理。\n"
                               "与上面的收边配合使用，单靠任一个都不够干净。"
                }),
                "edge_softness": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 4.0, "step": 0.05,
                    "tooltip": "边缘过渡宽度。原图边缘带抗锯齿，过渡太硬会有锯齿白边；\n"
                               "调大更柔和，调小更锐利。"
                }),
                "fragment_threshold": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "清掉面积不足主体这一比例的连通碎片。\n"
                               "网格切分会把相邻格子探过来的部件（手杖尖、飘起的衣角）\n"
                               "切进本格，既难看又会撑大自动裁剪的范围。\n"
                               "0 = 不清理；与身体相连的道具不会被误删。"
                }),
                "expand_beyond_cell": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "允许角色超出格子边界（强烈建议开启）。\n"
                               "关闭时按格子严格切分，角色只要探出格线就会被切断\n"
                               "——最常见的是脚、飘起的斗篷和手杖被削掉一截。\n"
                               "开启后格子只用来判定「这是哪个方向」，实际范围由角色\n"
                               "自身的连通区域决定；按质心归属，相邻角色不会被卷进来。\n"
                               "对三种 bg_mode 都有效：选「不处理」时也会在内部算一份\n"
                               "白底检测来圈定范围，输出仍保持不透明。"
                }),
                "auto_crop": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "按内容裁掉多余空白。\n"
                               "裁剪框取「该方向所有帧的并集」，因此整条序列尺寸一致，\n"
                               "既能合成视频，角色也不会在帧间跳动。"
                }),
                "crop_padding": ("INT", {
                    "default": 16, "min": 0, "max": 200, "step": 1,
                    "tooltip": "裁剪时在内容外保留的边距（像素）。\n"
                               "给足边距不仅好看，也给后续的描边、发光、\n"
                               "阴影等特效留出余量，免得贴着边显得像被切了。"
                }),
            },
            "optional": {
                "masks": ("MASK", {
                    "tooltip": "可选。已有的透明通道（如上游抠图结果），\n"
                               "配合 bg_mode=已带透明通道 使用。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",) * MAX_DIRS + ("STRING",)
    RETURN_NAMES = tuple(f"dir_{i + 1}" for i in range(MAX_DIRS)) + ("info",)
    FUNCTION = "split"
    CATEGORY = "Rui-Node🐶/图像调节🎨"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def split(self, images, grid_cols, grid_rows, empty_cells,
              direction_names, bg_mode, bg_threshold, edge_shrink,
              decontaminate, edge_softness, fragment_threshold,
              expand_beyond_cell, auto_crop, crop_padding, masks=None):
        mode = _BG_MODES.get(bg_mode) or _BG_ALIAS.get(bg_mode, "white")
        B, H, W, C = images.shape
        cols, rows = int(grid_cols), int(grid_rows)
        total = cols * rows
        empties = _parse_cells(empty_cells, total)
        cell_ids = [i for i in range(total) if i not in empties]
        n_dir = len(cell_ids)

        names = [s.strip() for s in str(direction_names).replace("，", ",").split(",")
                 if s.strip()]
        while len(names) < n_dir:
            names.append(f"dir{len(names) + 1}")

        arr = images.detach().cpu().float().numpy()
        if C == 4:
            rgb_all, a_in = arr[..., :3], arr[..., 3]
        else:
            rgb_all, a_in = arr[..., :3], None
        if masks is not None:
            m = masks.detach().cpu().float().numpy()
            if m.ndim == 2:
                m = m[None]
            if m.shape[0] != B:
                m = np.repeat(m[:1], B, axis=0)
            a_in = m

        # 格子边界按浮点等分再取整，避免整除不尽时累计误差（如 1112/3）
        ybnd = [int(round(r * H / rows)) for r in range(rows + 1)]
        xbnd = [int(round(c * W / cols)) for c in range(cols + 1)]

        pad = int(crop_padding)
        # 注意不要因为 bg_mode=不处理 就跳过这条路径：那样会退回严格格线切分，
        # 角色照样被切。定位与输出是两件事，分开处理即可。
        use_expand = bool(expand_beyond_cell) and _HAS_SCIPY

        def locate_alpha(b):
            """
            仅用于圈定角色范围的 alpha。
            即使用户选了「不处理（输出不透明）」，这里也要照算一份 ——
            否则无从判断角色到哪儿为止，只能按格线硬切。
            """
            if mode == "keep" and a_in is not None:
                return a_in[b]
            return _white_to_alpha(rgb_all[b], bg_threshold, edge_softness)

        def frame_alpha(b):
            """真正写进输出的 alpha。"""
            if mode == "none":
                return np.ones((H, W), dtype=np.float32)
            return locate_alpha(b)

        outs, notes = [], []

        if use_expand:
            # ===== 内容自适应：格子只定方向，范围由角色自身的连通块决定 =====
            # 第一遍只求包围盒（全图坐标），不留像素，避免整段序列驻留内存
            boxes = [None] * n_dir
            for b in range(B):
                lab, groups = _assign_blocks(locate_alpha(b), ybnd, xbnd,
                                             cols, fragment_threshold)
                if lab is None:
                    continue
                for di, cid in enumerate(cell_ids):
                    blk = groups.get(cid)
                    if not blk:
                        continue
                    bb = _bbox(np.isin(lab, blk).astype(np.float32), 0.5)
                    if bb is None:
                        continue
                    boxes[di] = bb if boxes[di] is None else (
                        min(boxes[di][0], bb[0]), max(boxes[di][1], bb[1]),
                        min(boxes[di][2], bb[2]), max(boxes[di][3], bb[3]))

            # 没有 auto_crop 时退回该格的格线范围，仍允许块超界的像素带出来
            final = []
            for di, cid in enumerate(cell_ids):
                r, c = divmod(cid, cols)
                if auto_crop and boxes[di] is not None:
                    y0, y1, x0, x1 = boxes[di]
                    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
                    y1 = min(H, y1 + pad); x1 = min(W, x1 + pad)
                elif boxes[di] is not None:
                    y0, y1, x0, x1 = (min(ybnd[r], boxes[di][0]),
                                      max(ybnd[r + 1], boxes[di][1]),
                                      min(xbnd[c], boxes[di][2]),
                                      max(xbnd[c + 1], boxes[di][3]))
                else:
                    y0, y1, x0, x1 = ybnd[r], ybnd[r + 1], xbnd[c], xbnd[c + 1]
                final.append((y0, y1, x0, x1))
                outs.append(np.zeros((B, y1 - y0, x1 - x0, 4), dtype=np.float32))

            # 第二遍按并集框提取；只保留归属本格的连通块，邻居不会混进来
            for b in range(B):
                la = locate_alpha(b)
                af = frame_alpha(b)
                lab, groups = _assign_blocks(la, ybnd, xbnd, cols, fragment_threshold)
                for di, cid in enumerate(cell_ids):
                    y0, y1, x0, x1 = final[di]
                    rgb = rgb_all[b, y0:y1, x0:x1]
                    blk = groups.get(cid) if groups else None
                    m = (np.isin(lab[y0:y1, x0:x1], blk) if blk
                         else np.zeros(rgb.shape[:2], dtype=bool))

                    if mode == "none":
                        # 不透明输出。裁剪框为了不切断角色必然会框进邻居的像素，
                        # 「原样保留」与「不切断」不可兼得 —— 这里把非本角色的部分
                        # 合成到白底，既保住完整的角色，也不会混进旁边那位。
                        a = (locate_alpha(b)[y0:y1, x0:x1] * m)[..., None]
                        outs[di][b, ..., :3] = rgb * a + (1.0 - a)
                        outs[di][b, ..., 3] = 1.0
                        continue

                    if blk:
                        ac = _shrink_alpha(af[y0:y1, x0:x1] * m, edge_shrink)
                        outs[di][b, ..., :3] = _decontaminate(rgb, ac, decontaminate)
                        outs[di][b, ..., 3] = ac
                    else:
                        outs[di][b, ..., :3] = rgb

            for di in range(n_dir):
                y0, y1, x0, x1 = final[di]
                notes.append(f"{di + 1}.{names[di]} 格{cell_ids[di]} "
                             f"{x1 - x0}×{y1 - y0}")
                outs[di] = torch.from_numpy(outs[di])
        else:
            # ===== 严格按格线切分（角色探出格线会被截断）=====
            per_dir = [[] for _ in range(n_dir)]
            boxes = [None] * n_dir
            for b in range(B):
                af = frame_alpha(b)
                for di, cid in enumerate(cell_ids):
                    r, c = divmod(cid, cols)
                    y0, y1, x0, x1 = ybnd[r], ybnd[r + 1], xbnd[c], xbnd[c + 1]
                    rgb = rgb_all[b, y0:y1, x0:x1]
                    alpha = af[y0:y1, x0:x1]
                    if mode != "none":
                        alpha = _drop_fragments(alpha, fragment_threshold)
                    per_dir[di].append((rgb, alpha))
                    if auto_crop:
                        bb = _bbox(alpha)
                        if bb is not None:
                            boxes[di] = bb if boxes[di] is None else (
                                min(boxes[di][0], bb[0]), max(boxes[di][1], bb[1]),
                                min(boxes[di][2], bb[2]), max(boxes[di][3], bb[3]))
            for di in range(n_dir):
                frames = per_dir[di]
                ch, cw = frames[0][0].shape[:2]
                if auto_crop and boxes[di] is not None:
                    y0, y1, x0, x1 = boxes[di]
                    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
                    y1 = min(ch, y1 + pad); x1 = min(cw, x1 + pad)
                else:
                    y0, y1, x0, x1 = 0, ch, 0, cw
                stack = np.empty((len(frames), y1 - y0, x1 - x0, 4), dtype=np.float32)
                for fi, (rgb, alpha) in enumerate(frames):
                    rc = rgb[y0:y1, x0:x1]
                    ac = alpha[y0:y1, x0:x1]
                    if mode != "none":
                        ac = _shrink_alpha(ac, edge_shrink)
                        rc = _decontaminate(rc, ac, decontaminate)
                    stack[fi, ..., :3] = rc
                    stack[fi, ..., 3] = ac
                outs.append(torch.from_numpy(stack))
                notes.append(f"{di + 1}.{names[di]} 格{cell_ids[di]} "
                             f"{x1 - x0}×{y1 - y0}")

        info = (f"输入 {B} 帧 {W}×{H} → {cols}×{rows} 网格，"
                f"空格 {sorted(empties) if empties else '无'}，"
                f"得到 {n_dir} 个方向 × {B} 帧\n" + " | ".join(notes))
        if n_dir > MAX_DIRS:
            info += f"\n⚠ 方向数 {n_dir} 超过输出口数量 {MAX_DIRS}，只输出前 {MAX_DIRS} 个"
        elif n_dir < MAX_DIRS:
            info += (f"\n⚠ 方向数 {n_dir} 少于输出口数量 {MAX_DIRS}，"
                     f"dir_{n_dir + 1}~dir_{MAX_DIRS} 为占位空图，请勿使用")
        print(f"[Ruinode-8Dir] {info}")

        # 输出口数量固定，方向不足时补占位图，避免下游拿到 None 直接报错
        blank = torch.zeros((1, 8, 8, 4), dtype=torch.float32)
        result = [outs[i] if i < len(outs) else blank for i in range(MAX_DIRS)]
        return tuple(result) + (info,)


NODE_CLASS_MAPPINGS = {
    "RuiEightDirSplit": RuiEightDirSplit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiEightDirSplit": "八方向序列拆分 / 8-Direction Sprite Split",
}
