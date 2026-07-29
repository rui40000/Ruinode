# -*- coding: utf-8 -*-
"""
Lucida 模型代码（内嵌自 https://huggingface.co/egeorcun/lucida ，MIT）
=====================================================================
Lucida 是 BiRefNet_HR 的微调版，专攻伪装物体、透明材质、文字/Logo、
VFX 光效与插画。模型实现（birefnet.py / BiRefNet_config.py）与上游逐字节一致。

内嵌而非用 trust_remote_code 在线加载的原因：
- trust_remote_code=True 会从 HuggingFace 拉取并执行远程 Python 代码，
  ComfyUI 场景下既不该联网，也不该在用户机器上执行随时可变的远程代码；
- 内嵌后版本固定、可离线、可审计。
"""
# ruff: noqa: F401

from .birefnet import BiRefNet
from .BiRefNet_config import BiRefNetConfig

__all__ = ["BiRefNet", "BiRefNetConfig"]
