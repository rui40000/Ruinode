# -*- coding: utf-8 -*-
"""
内置复古调色板
==============
只收录能确认取值的经典调色板，宁缺毋滥 —— 调色板值错了，
产出的美术资产就和目标机器/风格对不上，属于很难事后发现的错误。
灰阶类由代码等距生成，不存在记错的问题。
"""
import numpy as np


def _hex(*codes):
    return np.array([[int(c[i:i + 2], 16) for i in (0, 2, 4)] for c in codes],
                    dtype=np.float32)


# PICO-8 官方 16 色
PICO8 = _hex(
    "000000", "1D2B53", "7E2553", "008751", "AB5236", "5F574F", "C2C3C7",
    "FFF1E8", "FF004D", "FFA300", "FFEC27", "00E436", "29ADFF", "83769C",
    "FF77A8", "FFCCAA",
)

# 初代 Game Boy（DMG）四级绿
GAMEBOY = _hex("0F380F", "306230", "8BAC0F", "9BBC0F")

# 1-bit 黑白
ONEBIT = _hex("000000", "FFFFFF")


def _gray(n):
    v = np.linspace(0, 255, n, dtype=np.float32)
    return np.stack([v, v, v], axis=1)


PALETTES = {
    "PICO-8 (16色)": PICO8,
    "Game Boy (4色绿)": GAMEBOY,
    "黑白 1-bit (2色)": ONEBIT,
    "灰阶 4 级": _gray(4),
    "灰阶 8 级": _gray(8),
    "灰阶 16 级": _gray(16),
}

PALETTE_NAMES = list(PALETTES.keys())
