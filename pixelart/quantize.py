# -*- coding: utf-8 -*-
"""
调色板生成、颜色映射与抖动
==========================
像素游戏资产通常要求「颜色数受控」——不是为了压缩，而是风格本身的一部分，
也方便后续做整套素材的统一改色。

聚类与最近邻匹配都在 CIELAB 空间做：RGB 空间里的欧氏距离与人眼感受
相差很远（绿色区域被严重高估），直接在 RGB 里挑色会丢暗部层次。
"""
import numpy as np

EPS = 1e-8


# ------------------------------------------------------------------ 色彩空间
def rgb_to_lab(rgb):
    """rgb: (...,3) in [0,1] -> CIELAB（D65）。"""
    rgb = np.clip(rgb, 0.0, 1.0)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = lin @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(np.maximum(xyz, EPS)),
                 xyz / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])], axis=-1)


# ------------------------------------------------------------------ 调色板
def kmeans_palette(pixels, k, iters=24, seed=0, sample=20000):
    """
    在 LAB 空间做 k-means，返回 (k,3) RGB[0,1] 调色板。
    簇心取该簇像素的 RGB 均值 —— 免去 LAB->RGB 反变换，也保证结果一定在色域内。
    """
    rng = np.random.default_rng(seed)
    px = pixels.reshape(-1, 3)
    if px.shape[0] > sample:                      # 大图抽样，聚类结果几乎不变
        px = px[rng.choice(px.shape[0], sample, replace=False)]
    k = int(max(1, min(k, px.shape[0])))
    lab = rgb_to_lab(px)

    # k-means++ 初始化：随机播种容易把多个簇心挤在同一片颜色里
    centers = np.empty((k, 3), dtype=np.float32)
    centers[0] = lab[rng.integers(px.shape[0])]
    d2 = ((lab - centers[0]) ** 2).sum(1)
    for i in range(1, k):
        prob = d2 / (d2.sum() + EPS)
        centers[i] = lab[rng.choice(px.shape[0], p=prob)]
        d2 = np.minimum(d2, ((lab - centers[i]) ** 2).sum(1))

    labels = np.zeros(px.shape[0], dtype=np.int64)
    for _ in range(iters):
        dist = ((lab[:, None, :] - centers[None]) ** 2).sum(-1)
        new = dist.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for i in range(k):
            m = labels == i
            if m.any():
                centers[i] = lab[m].mean(0)

    out = np.zeros((k, 3), dtype=np.float32)
    for i in range(k):
        m = labels == i
        out[i] = px[m].mean(0) if m.any() else px[rng.integers(px.shape[0])]
    return np.clip(out, 0.0, 1.0)


def median_cut_palette(pixels, k):
    """PIL 的 median cut，速度快、结果稳定，作为 k-means 之外的备选。"""
    from PIL import Image

    arr = np.clip(pixels.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
    side = int(np.ceil(np.sqrt(arr.shape[0])))
    pad = np.zeros((side * side, 3), dtype=np.uint8)
    pad[:arr.shape[0]] = arr
    pad[arr.shape[0]:] = arr[-1] if arr.shape[0] else 0
    im = Image.fromarray(pad.reshape(side, side, 3), "RGB")
    q = im.quantize(colors=int(max(1, k)), method=Image.Quantize.MEDIANCUT)
    pal = np.array(q.getpalette()[:int(max(1, k)) * 3],
                   dtype=np.float32).reshape(-1, 3) / 255.0
    return np.clip(pal, 0.0, 1.0)


# ------------------------------------------------------------------ 抖动
def bayer_matrix(n):
    """递归生成 n×n（n 为 2 的幂）有序抖动阈值矩阵，取值 [0,1)。"""
    m = np.array([[0.0]], dtype=np.float32)
    size = 1
    while size < n:
        m = np.block([[4 * m, 4 * m + 2],
                      [4 * m + 3, 4 * m + 1]])
        size *= 2
    return m / (size * size)


def _nearest(lab_px, lab_pal):
    """逐像素找 LAB 距离最近的调色板项，分块算以免一次性开出巨大矩阵。"""
    n = lab_px.shape[0]
    out = np.empty(n, dtype=np.int64)
    step = max(1, int(4_000_000 / max(1, lab_pal.shape[0])))
    for i in range(0, n, step):
        chunk = lab_px[i:i + step]
        d = ((chunk[:, None, :] - lab_pal[None]) ** 2).sum(-1)
        out[i:i + step] = d.argmin(1)
    return out


def apply_palette(img, palette, dither="none", strength=1.0, seed=0):
    """
    把 img (H,W,3) float[0,1] 映射到 palette (k,3)，返回同形状图像。

    dither:
      none            直接取最近色，边界干净，像素画默认
      bayer2/4/8      有序抖动，规则网点，复古感强且可平铺
      floyd-steinberg 误差扩散，过渡最自然但纹理不规则
      noise           随机抖动，打散色带又不产生规则网点
    strength 是抖动幅度相对「调色板平均色距」的比例。
    """
    H, W = img.shape[:2]
    pal = np.clip(np.asarray(palette, dtype=np.float32), 0.0, 1.0)
    lab_pal = rgb_to_lab(pal)

    # 抖动幅度参照调色板里相邻颜色的典型间距，否则同一强度在
    # 4 色盘和 64 色盘上的观感天差地别
    if pal.shape[0] > 1:
        d = np.sqrt(((pal[:, None, :] - pal[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        amp = float(np.median(d.min(axis=1))) * float(strength)
    else:
        amp = 0.0

    work = img.astype(np.float32).copy()

    if dither.startswith("bayer") and amp > 0:
        n = int(dither[5:] or 4)
        m = bayer_matrix(n)
        tile = np.tile(m, (H // n + 1, W // n + 1))[:H, :W]
        work = work + ((tile - 0.5) * amp)[..., None]
    elif dither == "noise" and amp > 0:
        rng = np.random.default_rng(seed)
        work = work + (rng.random((H, W, 1), dtype=np.float32) - 0.5) * amp

    if dither == "floyd-steinberg":
        # 误差扩散必须串行；像素画输出通常很小（几十到几百像素），
        # 这点循环开销可以接受
        out = np.empty((H, W, 3), dtype=np.float32)
        buf = work
        for y in range(H):
            for x in range(W):
                old = buf[y, x]
                i = int(_nearest(rgb_to_lab(old[None]), lab_pal)[0])
                new = pal[i]
                out[y, x] = new
                err = old - new
                if x + 1 < W:
                    buf[y, x + 1] += err * (7 / 16)
                if y + 1 < H:
                    if x > 0:
                        buf[y + 1, x - 1] += err * (3 / 16)
                    buf[y + 1, x] += err * (5 / 16)
                    if x + 1 < W:
                        buf[y + 1, x + 1] += err * (1 / 16)
        return out

    idx = _nearest(rgb_to_lab(np.clip(work, 0, 1).reshape(-1, 3)), lab_pal)
    return pal[idx].reshape(H, W, 3)
