# -*- coding: utf-8 -*-
"""
ZenMux 模型注册表
=================
从随包分发的 models_snapshot.json 加载模型清单，为节点提供：

- all_vendors()          厂商列表（模型 id 的前缀，如 "openai"）
- all_model_labels()     全量「带价格」下拉标签，按厂商 → 模型排序
- default_model_label()  默认模型 openai/gpt-5.4-nano 对应的标签
- label_to_model_id()    从下拉标签解析真实 model id（对旧标签/纯 id 也兼容）

标签格式（价格单位：USD / 百万 token）：
    openai/gpt-5.4-nano [入$0.04/M 出$0.32/M]
标签第一个空格前恒为 model id，前端级联 JS 与后端解析都依赖这一点。

快照可用 build_snapshot.py 随时重新拉取更新。
"""
import json
import os

DEFAULT_MODEL_ID = "openai/gpt-5.4-nano"

_SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "models_snapshot.json")


def _fmt_price(v):
    """10.0 -> '10'，0.04 -> '0.04'，None -> '?'。"""
    if v is None:
        return "?"
    try:
        s = f"{float(v):.4f}".rstrip("0").rstrip(".")
        return s if s else "0"
    except (TypeError, ValueError):
        return "?"


def _make_label(m):
    return (f"{m['id']} "
            f"[入${_fmt_price(m.get('input_price'))}/M "
            f"出${_fmt_price(m.get('output_price'))}/M]")


def _load_models():
    """读取快照；文件缺失/损坏时退化为只含默认模型的最小清单。"""
    try:
        with open(_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            models = json.load(f)["models"]
        if models:
            return models
    except Exception as e:
        print(f"[Rui-Node] ZenMux 模型快照加载失败（{e}），退化为最小模型清单。"
              f"可运行 zenmux/build_snapshot.py 重新生成。")
    return [{
        "id": DEFAULT_MODEL_ID,
        "vendor": DEFAULT_MODEL_ID.split("/")[0],
        "input_price": None,
        "output_price": None,
    }]


# 模块加载时构建一次即可（ComfyUI 启动时构建节点定义）
_MODELS = _load_models()
_LABELS = [_make_label(m) for m in _MODELS]
_LABEL_TO_ID = {lb: m["id"] for lb, m in zip(_LABELS, _MODELS)}
_KNOWN_IDS = {m["id"] for m in _MODELS}
_VENDORS = sorted({m.get("vendor") or m["id"].split("/")[0] for m in _MODELS},
                  key=str.lower)


def all_vendors():
    return list(_VENDORS)


def all_model_labels():
    return list(_LABELS)


def default_model_label():
    for lb, m in zip(_LABELS, _MODELS):
        if m["id"] == DEFAULT_MODEL_ID:
            return lb
    return _LABELS[0]


def label_to_model_id(label):
    """
    从下拉标签解析 model id。三层兼容：
    1. 当前标签精确命中；
    2. 旧版工作流里存的标签（价格已变动）→ 取第一个空格前的 id 段；
    3. 用户直接填了纯 model id → 原样返回。
    解析不出（空串等）返回 None，由调用方兜底到默认模型。
    """
    s = (label or "").strip()
    if not s:
        return None
    if s in _LABEL_TO_ID:
        return _LABEL_TO_ID[s]
    head = s.split(" ")[0].split("\t")[0]
    if head in _KNOWN_IDS:
        return head
    # 未收录的 id（快照偏旧但模型真实存在）也放行，交给服务端判定
    return head if head else None
