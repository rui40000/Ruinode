# -*- coding: utf-8 -*-
"""
SDMatte 权重读取。

官方发布的 SDMatte_plus.pth 是 detectron2 的训练检查点，顶层结构为
    {"model": {...1316 个张量...}, "trainer": {...优化器状态...}, "iteration": int}
其中 trainer 约占 6GB，推理完全用不到，且内部引用了 omegaconf 的类型，
直接 torch.load 会连带反序列化它、白白吃掉一倍内存，还平添一个依赖。

因此这里自己解析 zip 容器：先用受限的 Unpickler 还原出对象骨架（张量一律替换成
惰性占位），再只把 model 段的张量真正读进内存。附带收益是全程不执行 pickle 里的
任意代码，比 torch.load(weights_only=False) 更安全。
"""

import io
import os
import pickle
import zipfile
from collections import OrderedDict

import torch

# torch storage 类名 -> dtype
_STORAGE_DTYPES = {
    "FloatStorage": torch.float32,
    "HalfStorage": torch.float16,
    "DoubleStorage": torch.float64,
    "BFloat16Storage": torch.bfloat16,
    "LongStorage": torch.int64,
    "IntStorage": torch.int32,
    "ShortStorage": torch.int16,
    "CharStorage": torch.int8,
    "ByteStorage": torch.uint8,
    "BoolStorage": torch.bool,
}


class _Dummy:
    """吞掉 trainer 段里那些我们不关心的对象（omegaconf 容器等）。"""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

    def __setitem__(self, key, value):
        pass

    def append(self, *args):
        pass

    def __reduce__(self):
        return (_Dummy, ())


class _LazyStorage:
    __slots__ = ("key", "dtype", "numel")

    def __init__(self, key, dtype, numel):
        self.key = key
        self.dtype = dtype
        self.numel = numel


class _LazyTensor:
    """记下重建一个张量所需的全部信息，但不读取数据。"""

    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = size
        self.stride = stride


def _lazy_rebuild(storage, storage_offset, size, stride, *args, **kwargs):
    return _LazyTensor(storage, storage_offset, tuple(size), tuple(stride))


class _StorageType:
    """代表 torch.FloatStorage 之类的类对象，只用于查 dtype。"""

    def __init__(self, name):
        self.dtype = _STORAGE_DTYPES.get(name, torch.float32)


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _lazy_rebuild
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        # 其余一律替换成惰性替身，绝不 import 外部模块、绝不执行其代码
        return _Dummy

    def persistent_load(self, saved_id):
        # 形如 ('storage', <storage_type>, key, location, numel)
        if not (isinstance(saved_id, tuple) and len(saved_id) >= 5 and saved_id[0] == "storage"):
            return _Dummy()
        storage_type, key, location, numel = saved_id[1], saved_id[2], saved_id[3], saved_id[4]
        dtype = getattr(storage_type, "dtype", torch.float32)
        return _LazyStorage(str(key), dtype, numel)


def _materialize(zf, prefix, lazy, name):
    """把一个 _LazyTensor 真正读成 torch.Tensor。"""
    st = lazy.storage
    raw = zf.read(f"{prefix}data/{st.key}")
    # frombuffer 需要可写缓冲区，bytearray 在此产生唯一一次拷贝
    flat = torch.frombuffer(bytearray(raw), dtype=st.dtype)
    if not lazy.size:  # 0 维标量
        return flat[lazy.offset].clone()
    return torch.as_strided(flat, lazy.size, lazy.stride, lazy.offset).clone()


def _load_pth_model_only(path):
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        pkl_name = next((n for n in names if n.endswith("data.pkl")), None)
        if pkl_name is None:
            raise ValueError(f"不是 torch 的 zip 检查点：{path}")
        prefix = pkl_name[: -len("data.pkl")]

        # 字节序校验：权重按原始字节读入，端序不符会得到垃圾数据
        if f"{prefix}byteorder" in names:
            bo = zf.read(f"{prefix}byteorder").decode().strip()
            if bo != "little":
                raise ValueError(f"检查点字节序为 {bo}，本加载器仅支持小端")

        skeleton = _RestrictedUnpickler(io.BytesIO(zf.read(pkl_name))).load()

        if isinstance(skeleton, dict) and isinstance(skeleton.get("model"), dict):
            raw_sd = skeleton["model"]
        elif isinstance(skeleton, dict) and isinstance(skeleton.get("state_dict"), dict):
            raw_sd = skeleton["state_dict"]
        elif isinstance(skeleton, dict):
            raw_sd = skeleton
        else:
            raise ValueError(f"无法识别的检查点结构：{type(skeleton)}")

        state_dict = {}
        for k, v in raw_sd.items():
            if isinstance(v, _LazyTensor):
                state_dict[k] = _materialize(zf, prefix, v, k)
        return state_dict


def load_sdmatte_state_dict(path):
    """读入 SDMatte 权重，返回纯张量的 state_dict。支持 .pth 与 .safetensors。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    if ext in (".pth", ".pt", ".ckpt", ".bin"):
        return _load_pth_model_only(path)
    raise ValueError(f"不支持的权重格式：{ext}")


_TM_PREFIX = "text_encoder.text_model."
_TE_PREFIX = "text_encoder."


def adapt_state_dict_to_model(state_dict, model):
    """
    抹平 transformers 版本差异导致的 text_encoder 键名错位。

    官方权重用 transformers 4.x 保存，CLIPTextModel 内部裹了一层 text_model，
    键形如 text_encoder.text_model.encoder.layers.0...；
    transformers 5.x 起该包装层被移除，键变成 text_encoder.encoder.layers.0...。

    两者不匹配时 load_state_dict(strict=False) 会把 text_encoder 的 372 个权重
    悄悄全部丢掉、让它停留在随机初始化状态 —— 不报错，但输出质量明显劣化。
    这里按当前环境的实际结构做一次前缀增删。
    """
    model_keys = set(model.state_dict().keys())
    model_wraps = any(k.startswith(_TM_PREFIX) for k in model_keys)
    ckpt_wraps = any(k.startswith(_TM_PREFIX) for k in state_dict)

    if model_wraps == ckpt_wraps:
        return state_dict

    adapted = {}
    if ckpt_wraps and not model_wraps:
        # 权重带 text_model，当前 transformers 不带 -> 剥离
        for k, v in state_dict.items():
            if k.startswith(_TM_PREFIX):
                adapted[_TE_PREFIX + k[len(_TM_PREFIX):]] = v
            else:
                adapted[k] = v
        print(f"[Ruinode-SDMatte] 已适配 text_encoder 键名（剥离 text_model 层，transformers>=5）")
    else:
        # 权重不带 text_model，当前 transformers 需要 -> 补上
        for k, v in state_dict.items():
            if k.startswith(_TE_PREFIX) and not k.startswith(_TM_PREFIX):
                adapted[_TM_PREFIX + k[len(_TE_PREFIX):]] = v
            else:
                adapted[k] = v
        print(f"[Ruinode-SDMatte] 已适配 text_encoder 键名（补回 text_model 层，transformers<5）")
    return adapted
