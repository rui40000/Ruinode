# -*- coding: utf-8 -*-
"""
SDMatte 推理代码（移植自 vivoCameraResearch/SDMatte，MIT License）。

此处刻意不在包级别导入 meta_arch —— 它依赖 diffusers/transformers，
若环境缺失会让整个 Ruinode 节点包加载失败。改由 sdmatte_node 在真正用到时再导入。
"""
