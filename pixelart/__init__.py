# -*- coding: utf-8 -*-
"""
像素化算法（Ruinode 自研，纯 numpy/PIL 实现）
=============================================
面向像素游戏资产制作，而非「马赛克滤镜」：产出的是真实小分辨率、
颜色数受控、边缘硬朗、可直接用作 sprite 的图。

- grid     像素网格检测（含八度/内容周期两类误判的对策）与按网格降采样
- quantize CIELAB 空间的调色板生成、颜色映射与四种抖动
- palettes 内置复古调色板
"""
# ruff: noqa: F401

from .grid import detect_grid, downsample_grid
from .palettes import PALETTES, PALETTE_NAMES
from .quantize import (apply_palette, kmeans_palette, median_cut_palette,
                       rgb_to_lab)

__all__ = [
    "detect_grid", "downsample_grid",
    "kmeans_palette", "median_cut_palette", "apply_palette", "rgb_to_lab",
    "PALETTES", "PALETTE_NAMES",
]
