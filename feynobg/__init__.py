# -*- coding: utf-8 -*-
"""
FeyNobg 推理子集（内嵌自 https://github.com/feyninc/nobg ，Apache-2.0）
=====================================================================
只保留推理必需的部分，供 Ruinode 的 FeyNobg 抠图节点使用：

- modeling_birefnet.py / loss.py / mixin.py / utils.py —— 与上游逐字节一致
- image_processing_birefnet.py —— 已重写，去掉对 transformers>=5.4 的依赖
  （上游继承 TorchvisionBackend，在 transformers 4.x 上直接 ImportError）

上游的 auto.py 未内嵌：其 AutoModel.from_pretrained 会调 model_info() 联网查
仓库 tags，而 ComfyUI 场景下模型是本地目录，联网检查既没必要又会在离线时失败。
节点改为直接调用 BiRefNet.from_pretrained(本地目录)。
"""
# ruff: noqa: F401

from .birefnet.image_processing_birefnet import BiRefNetImageProcessor
from .birefnet.modeling_birefnet import BiRefNet, BiRefNetConfig

# 对应上游 nobg 版本
__version__ = "0.2.4"
__all__ = ["BiRefNet", "BiRefNetConfig", "BiRefNetImageProcessor"]
