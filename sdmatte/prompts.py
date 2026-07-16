# -*- coding: utf-8 -*-
"""
视觉提示构造。

SDMatte 是交互式抠图：除了原图，还要一个"指哪抠哪"的视觉提示（框/掩码/点），
以及该提示的归一化坐标。官方在 data/dataset.py 里用 GenBBox / GenMask / GenPoint
从 GT alpha 现场生成这些提示；在 ComfyUI 里没有 GT，改由用户给的 mask 充当提示来源，
这正是官方设计的交互用法。

除此之外，各函数的行为与官方测试期逐行对齐（含 1024 分辨率、连通域筛选、
sigma=radius 的高斯点扩散、以及 x2-1 归一化），偏离任何一处都会让结果偏离论文指标。
"""

import cv2
import numpy as np
import scipy.ndimage
from scipy.ndimage import label

# 官方 configs/SDMatte.py: psm="gauss", radius=25；测试期用 radius + 10
DEFAULT_POINT_RADIUS = 35
DEFAULT_POINT_THRES = 0.8
NUM_POINTS = 10


def resize_image(img, size):
    """等价于官方 Resize：双线性缩放到 size x size。"""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def gen_bbox(ref, coe_scale=0.0, rng=None):
    """
    复刻官方 GenBBox（测试期 coe_scale=0，即不做随机扰动）。

    返回矩形提示掩码与归一化坐标 [x_min, y_min, x_max, y_max]。
    当存在一个显著大于其余的连通域时，官方只取该主体的外接框，
    以免零星噪点把框撑大。
    """
    height, width = ref.shape
    coords = np.nonzero(ref)
    if coords[0].size == 0 or coords[1].size == 0:
        return np.zeros_like(ref, dtype=np.float32), np.array([0, 0, 1, 1], dtype=np.float32)

    binary_mask = ref > 0
    labeled_array, num_features = label(binary_mask)
    y_min, x_min = np.argwhere(binary_mask).min(axis=0)
    y_max, x_max = np.argwhere(binary_mask).max(axis=0)

    if num_features > 0:
        component_coords = [np.argwhere(labeled_array == i) for i in range(1, num_features + 1)]
        areas = [c.shape[0] for c in component_coords]
        sorted_areas_idx = np.argsort(areas)[::-1]
        max_area_idx = sorted_areas_idx[0]
        second_max_area_idx = sorted_areas_idx[1] if len(sorted_areas_idx) > 1 else None
        max_area = areas[max_area_idx]
        second_max_area = areas[second_max_area_idx] if second_max_area_idx is not None else 0
        if max_area >= 10 * second_max_area:
            max_coords = component_coords[max_area_idx]
            y_min, x_min = max_coords.min(axis=0)
            y_max, x_max = max_coords.max(axis=0)

    if coe_scale:
        rng = rng or np.random
        coe = rng.uniform(0, coe_scale)
        padding_y = int(coe * (y_max - y_min))
        padding_x = int(coe * (x_max - x_min))
        y_min_p = padding_y if rng.random() < 0.5 else -padding_y
        y_max_p = padding_y if rng.random() < 0.5 else -padding_y
        x_min_p = padding_x if rng.random() < 0.5 else -padding_x
        x_max_p = padding_x if rng.random() < 0.5 else -padding_x
        y_min, y_max = max(0, y_min + y_min_p), min(height, y_max + y_max_p)
        x_min, x_max = max(0, x_min + x_min_p), min(width, x_max + x_max_p)

    bbox_mask = np.zeros_like(ref, dtype=np.float32)
    bbox_mask[y_min:y_max, x_min:x_max] = 1

    coords_norm = np.array(
        [x_min / width, y_min / height, x_max / width, y_max / height], dtype=np.float32
    )
    return bbox_mask, coords_norm


def gen_mask(ref):
    """复刻官方 GenMask 在"掩码已给定"时的分支：掩码原样，坐标取其外接框。"""
    h, w = ref.shape
    coords = np.nonzero(ref)
    if coords[0].size == 0 or coords[1].size == 0:
        mask_coords = np.array([0, 0, 1, 1], dtype=np.float32)
    else:
        y_min, x_min = np.argwhere(ref).min(axis=0)
        y_max, x_max = np.argwhere(ref).max(axis=0)
        mask_coords = np.array([x_min / w, y_min / h, x_max / w, y_max / h], dtype=np.float32)
    return ref.astype(np.float32), mask_coords


def gen_point(ref, thres=DEFAULT_POINT_THRES, radius=DEFAULT_POINT_RADIUS, seed=0):
    """
    复刻官方 GenPoint 的 gauss 模式：在提示区域内随机取 10 个点，
    每点摊开成一个 sigma=radius 的高斯斑，取逐像素最大值合并。

    官方用的是全局 np.random，这里换成带种子的独立发生器，让结果可复现。
    """
    height, width = ref.shape
    alpha_mask = (ref > thres).astype(np.float32)
    y_coords, x_coords = np.where(alpha_mask == 1)

    if len(y_coords) < NUM_POINTS:
        return np.zeros_like(ref, dtype=np.float32), np.zeros(20, dtype=np.float32)

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(y_coords), size=NUM_POINTS, replace=False)

    point_mask = np.zeros_like(ref, dtype=np.float32)
    point_coords = []
    for idx in selected_indices:
        y_center = y_coords[idx]
        x_center = x_coords[idx]
        tmp_mask = np.zeros_like(ref, dtype=np.float32)
        tmp_mask[y_center, x_center] = 1
        tmp_mask = scipy.ndimage.gaussian_filter(tmp_mask, sigma=radius)
        peak = np.max(tmp_mask)
        if peak > 0:
            tmp_mask = tmp_mask / peak
        point_mask = np.maximum(point_mask, tmp_mask)
        point_coords.append(x_center / width)
        point_coords.append(y_center / height)

    if len(point_coords) < 20:
        point_coords = np.concatenate([point_coords, np.zeros(20 - len(point_coords))])

    return point_mask, np.array(point_coords[:20], dtype=np.float32)


def gen_auto(ref):
    """auto 模式：整幅图都是提示区域，不提供任何定位信息。"""
    return np.ones_like(ref, dtype=np.float32), np.array([0, 0, 1, 1], dtype=np.float32)


def build_prompt(ref, prompt_type, point_radius=DEFAULT_POINT_RADIUS, seed=0):
    """
    按提示类型产出 (提示掩码, 坐标)。

    ref 为 [H, W] 的 float32，取值 [0, 1]，且已缩放到推理分辨率。
    """
    if prompt_type == "bbox_mask":
        return gen_bbox(ref)
    if prompt_type == "mask":
        return gen_mask(ref)
    if prompt_type == "point_mask":
        return gen_point(ref, radius=point_radius, seed=seed)
    if prompt_type == "auto_mask":
        return gen_auto(ref)
    raise ValueError(f"未知的提示类型：{prompt_type}")


def normalize(x):
    """官方 Normalize：把 [0,1] 映射到 [-1,1]。"""
    return x.astype(np.float32) * 2 - 1
