# -*- coding: utf-8 -*-
"""
像素网格检测与按网格降采样
==========================

两件事：
1) detect_grid —— 猜出「这张图其实是多少倍放大的像素图」，返回每轴的
   单元大小与相位。用于把 AI 生成的模糊/歪斜伪像素图还原成真正的像素图。
2) downsample_grid —— 按给定网格把图切成单元，每个单元取一个代表色。

检测思路：真网格的特征是「单元边界处颜色跳变、单元内部几乎不变」。
对每个候选单元大小 s 与相位 p，比较落在边界上的行/列差异均值与
落在内部的差异均值，比值越大越像真网格。

两个坑（业界公认，naive 自相关最容易栽在这）：
- **谐波/八度错误**：2s 与 s 得分往往接近，容易把 2 倍大小当成真值。
  故取得最高分后，回头检查其真约数，若某个约数得分不显著更差，
  说明它才是基频（octave killer）。
- **内容尺度冒充像素尺度**：画面里重复的纹理/花纹也会形成周期。
  真网格对相位极其敏感（对错相位分数暴跌），内容周期则不敏感，
  故把「最佳相位与最差相位的分差」并入评分（anti-phase 项）。

普通照片没有网格可言，检测分数会很低，由 min_confidence 判定为「未检出」。
"""
import numpy as np

EPS = 1e-6


def _axis_profile(img, axis, max_lines=96):
    """
    取出用于分析该轴周期的采样面。

    检测 x 方向周期时只需要若干代表性的行（而非全部），
    抽样后计算量与图像高度脱钩，1024² 的图也是毫秒级。
    返回 (L, N, C)：N 是该轴长度。
    """
    if axis == 1:
        h = img.shape[0]
        idx = np.linspace(0, h - 1, min(h, max_lines)).astype(np.int64)
        return img[idx]
    w = img.shape[1]
    idx = np.linspace(0, w - 1, min(w, max_lines)).astype(np.int64)
    return img[:, idx].transpose(1, 0, 2)


def _prefix(prof):
    """沿分析轴的前缀和与平方前缀和，之后任意分组的均值/方差都是 O(1)。"""
    L, N, C = prof.shape
    z = np.zeros((L, 1, C), dtype=np.float64)
    p = prof.astype(np.float64)
    return (np.concatenate([z, p.cumsum(1)], axis=1),
            np.concatenate([z, (p * p).cumsum(1)], axis=1))


def _score_phase(S, S2, N, s, p):
    """
    评估「单元大小 s、相位 p」这组假设有多像真网格。

    判据是两件事同时成立：
      单元内部要均匀   -> 组内方差 V 小
      相邻单元要有跳变 -> 组间均值差 D 大
    取 D / sqrt(V) 作为分数。

    为什么用组内方差而不是相邻像素差分：差分对模糊极其敏感，
    AI 生成的伪像素图边界都带抗锯齿，尖峰被摊平后就压不住画面里的
    内容周期（实测会把 4 像素的网格判成 24~28）。而组内方差天然反向
    惩罚过大的 s —— 单元一旦跨过多个真实色块，方差必然爆掉。
    """
    starts = np.arange(p, N - s + 1, s, dtype=np.int64)
    if starts.size < 2:
        return -1.0
    ends = starts + s
    tot = S[:, ends] - S[:, starts]
    tot2 = S2[:, ends] - S2[:, starts]
    mean = tot / s
    var = np.maximum(tot2 / s - mean * mean, 0.0)
    v = float(np.sqrt(var.mean()))
    d = float(np.abs(np.diff(mean, axis=1)).mean())
    return d / (v + EPS)


def detect_axis(prof, min_size=2, max_size=64, octave_ratio=0.72):
    """返回该轴的 (单元大小, 相位, 置信度)。"""
    N = prof.shape[1]
    hi = max(min_size, min(max_size, N // 3))
    if hi < min_size:
        return 1, 0, 0.0
    S, S2 = _prefix(prof)

    scores = {}
    for s in range(min_size, hi + 1):
        best, best_p, worst = -1.0, 0, float("inf")
        for p in range(s):
            c = _score_phase(S, S2, N, s, p)
            if c < 0:
                continue
            if c > best:
                best, best_p = c, p
            worst = min(worst, c)
        if best <= 0:
            continue
        # anti-phase：真网格挪错相位分数会塌掉，内容周期则无所谓相位。
        # 把这个差异计入总分，用来甄别「像周期」和「真网格」。
        spread = best - (worst if worst < float("inf") else best)
        scores[s] = (best * (1.0 + spread / (best + EPS)), best, best_p)

    if not scores:
        return 1, 0, 0.0

    s_best = max(scores, key=lambda k: scores[k][0])
    # octave killer：谐波（2s、3s…）得分常与基频接近，
    # 若某个真约数分数不显著更差，那它才是真正的基频
    for s2 in sorted(d for d in scores if d < s_best and s_best % d == 0):
        if scores[s2][0] >= scores[s_best][0] * octave_ratio:
            s_best = s2
            break
    _, raw, phase = scores[s_best]
    return s_best, phase, raw


def detect_grid(img, min_size=2, max_size=64, min_confidence=2.0):
    """
    img: (H, W, 3) float [0,1]
    返回 dict(size_x, size_y, off_x, off_y, conf_x, conf_y, detected)

    conf 是「边界差异 / 内部差异」的比值：完美放大的像素图会很高（内部差异
    近乎为零），普通照片接近 1。
    """
    sx, px, cx = detect_axis(_axis_profile(img, 1), min_size, max_size)
    sy, py, cy = detect_axis(_axis_profile(img, 0), min_size, max_size)
    # 相位 p 表示「新单元的第一列/行」，直接就是切块起点
    return {
        "size_x": sx, "size_y": sy,
        "off_x": px % sx if sx > 1 else 0,
        "off_y": py % sy if sy > 1 else 0,
        "conf_x": cx, "conf_y": cy,
        "detected": bool(min(cx, cy) >= min_confidence and max(sx, sy) > 1),
    }


def _mode_colors(blocks):
    """
    blocks: (N, P, C) -> (N, C)
    取每个单元的主导色：先把颜色粗量化到 5bit/通道统计众数，
    再对落在众数 bin 里的原始像素求均值。
    直接用均值会凭空造出新颜色并糊掉边缘，这正是像素画最忌讳的。
    """
    N, P, C = blocks.shape
    q = np.clip((blocks * 31.0).astype(np.int32), 0, 31)
    if C >= 3:
        key = (q[..., 0] << 10) | (q[..., 1] << 5) | q[..., 2]
    else:
        key = q[..., 0]

    order = np.argsort(key, axis=1)
    ks = np.take_along_axis(key, order, axis=1)
    same = np.zeros_like(ks, dtype=bool)
    same[:, 1:] = ks[:, 1:] == ks[:, :-1]
    segid = np.cumsum(~same, axis=1) - 1            # 每个位置所属的段

    rows = np.repeat(np.arange(N), P)
    counts = np.bincount(rows * P + segid.ravel(),
                         minlength=N * P).reshape(N, P)
    best_seg = counts.argmax(axis=1)                # 最长的段 = 众数
    pos = np.argmax(segid == best_seg[:, None], axis=1)
    best_key = ks[np.arange(N), pos]

    mask = key == best_key[:, None]                 # (N,P)
    w = mask.sum(axis=1)[:, None].astype(np.float32)
    return (blocks * mask[..., None]).sum(axis=1) / np.maximum(w, 1.0)


def downsample_grid(img, size_x, size_y, off_x=0, off_y=0, method="dominant"):
    """
    按网格切块降采样。img: (H,W,C) float -> (rows, cols, C)

    method:
      dominant 每块的主导色（像素画首选，不会造出新颜色）
      median   逐通道中位数
      mean     均值（会产生新颜色、边缘发糊，仅在想要柔和过渡时用）
      center   取块中心像素（等价最近邻，最锐但受噪点影响）
    """
    H, W = img.shape[:2]
    size_x = max(1, int(size_x))
    size_y = max(1, int(size_y))
    off_x = int(off_x) % size_x if size_x > 1 else 0
    off_y = int(off_y) % size_y if size_y > 1 else 0

    cols = (W - off_x) // size_x
    rows = (H - off_y) // size_y
    if cols < 1 or rows < 1:
        # 单元比图还大：整图退化成一个像素
        return img.mean(axis=(0, 1))[None, None, :]

    crop = img[off_y:off_y + rows * size_y, off_x:off_x + cols * size_x]
    C = crop.shape[2]
    blk = crop.reshape(rows, size_y, cols, size_x, C).transpose(0, 2, 1, 3, 4)

    if method == "center":
        return blk[:, :, size_y // 2, size_x // 2, :].copy()

    flat = blk.reshape(rows * cols, size_y * size_x, C)
    if method == "mean":
        out = flat.mean(axis=1)
    elif method == "median":
        out = np.median(flat, axis=1)
    else:
        out = _mode_colors(flat)
    return out.reshape(rows, cols, C)
