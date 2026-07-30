# -*- coding: utf-8 -*-
"""
越光模型注册表
==============
越光的可用模型与价格由官方规范文档给定（不像 ZenMux 有可枚举的模型接口），
因此这里直接内置清单，不做在线快照 —— 少一个联网环节，也不会因为
拉取失败而让下拉变空。价格有变动时改这张表即可。

对外接口与 zenmux/model_registry.py 保持一致，便于两个节点共用同一套写法：
- all_model_labels()     带价格的下拉标签，按厂商分组、组内保持官方顺序
- default_model_label()  默认模型对应的标签
- label_to_model_id()    从标签解析真实 model id（对旧标签/纯 id 也兼容）
- model_prices()         按 model id 查（输入单价, 输出单价），USD/百万 token
- model_label_by_id()    按 model id 生成带价签标签

标签格式（价格单位：USD / 百万 token）：
    gpt-4o [入$2.5/M 出$10/M]
标签第一个空格前恒为 model id，后端解析依赖这一点。

注意：越光的 model id **不带厂商前缀**（是 gpt-4o 而非 openai/gpt-4o），
这点与 ZenMux 不同。下拉里同厂商的模型是靠排序聚在一起的，
搜索时输 gpt / claude / deepseek / kimi 即可过滤。
"""

# (厂商, model id, 输入单价, 输出单价)  单价单位：USD / 百万 token
# 数据来源：越光 API 接入规范（2026-07-29），docs.stars-cloud.com/cn
_MODELS_RAW = [
    # ---- OpenAI ----
    ("OpenAI", "gpt-5.6-sol", 4.75, 5.00),
    ("OpenAI", "gpt-5.6-terra", 2.375, 2.50),
    ("OpenAI", "gpt-5.6-luna", 0.95, 1.00),
    ("OpenAI", "gpt-4.1", 2.00, 8.00),
    ("OpenAI", "gpt-4.1-mini", 0.40, 1.60),
    ("OpenAI", "gpt-4o", 2.50, 10.00),
    ("OpenAI", "gpt-4o-mini", 0.15, 0.60),
    ("OpenAI", "o4-mini", 1.10, 4.40),
    ("OpenAI", "o3-mini", 1.10, 4.40),
    # ---- Anthropic ----
    ("Anthropic", "claude-opus-5", 5.00, 5.00),
    ("Anthropic", "claude-opus-4-7", 15.00, 75.00),
    ("Anthropic", "claude-opus-4-6", 15.00, 75.00),
    ("Anthropic", "claude-sonnet-5", 2.00, 2.00),
    ("Anthropic", "claude-sonnet-4-6", 3.00, 15.00),
    ("Anthropic", "claude-haiku-4-5-20251001", 0.80, 4.00),
    ("Anthropic", "claude-fable-5", 3.00, 15.00),
    # ---- DeepSeek ----
    ("DeepSeek", "deepseek-v4-pro", 2.19, 8.76),
    ("DeepSeek", "deepseek-v4-flash", 0.10, 0.30),
    ("DeepSeek", "deepseek-r1-250528", 0.55, 2.19),
    ("DeepSeek", "deepseek-v3-250324", 0.27, 1.10),
    # ---- Kimi (Moonshot) ----
    ("Kimi", "kimi-k3", 2.86, 2.86),
    ("Kimi", "kimi-k2.7-code", 1.00, 4.00),
    ("Kimi", "kimi-k2.6", 1.00, 4.00),
    ("Kimi", "kimi-k2.5", 1.00, 4.00),
    ("Kimi", "kimi-k2-thinking", 1.00, 4.00),
]

# 默认用最便宜的 deepseek-v4-flash（$0.1/$0.3）：官方示例也用它，
# 且默认值便宜可以避免误触发时产生意外费用。
DEFAULT_MODEL_ID = "deepseek-v4-flash"


def _fmt_price(v):
    """10.0 -> '10'，0.04 -> '0.04'，None -> '?'。"""
    if v is None:
        return "?"
    try:
        s = f"{float(v):.4f}".rstrip("0").rstrip(".")
        return s if s else "0"
    except (TypeError, ValueError):
        return "?"


_MODELS = [{"vendor": v, "id": i, "input_price": ip, "output_price": op}
           for v, i, ip, op in _MODELS_RAW]


def _make_label(m):
    return (f"{m['id']} "
            f"[入${_fmt_price(m.get('input_price'))}/M "
            f"出${_fmt_price(m.get('output_price'))}/M]")


_LABELS = [_make_label(m) for m in _MODELS]
_LABEL_TO_ID = {lb: m["id"] for lb, m in zip(_LABELS, _MODELS)}
_KNOWN_IDS = {m["id"] for m in _MODELS}
_ID_TO_MODEL = {m["id"]: m for m in _MODELS}


def all_model_labels():
    return list(_LABELS)


def default_model_label():
    for lb, m in zip(_LABELS, _MODELS):
        if m["id"] == DEFAULT_MODEL_ID:
            return lb
    return _LABELS[0]


def model_prices(model_id):
    """按 model id 查（输入单价, 输出单价），USD/百万 token；未知模型 (None, None)。"""
    m = _ID_TO_MODEL.get(model_id)
    if not m:
        return (None, None)
    return (m.get("input_price"), m.get("output_price"))


def model_label_by_id(model_id):
    """按 model id 生成带价签标签；未收录的 id 原样返回。"""
    m = _ID_TO_MODEL.get(model_id)
    return _make_label(m) if m else (model_id or "?")


def model_vendor(model_id):
    m = _ID_TO_MODEL.get(model_id)
    return m["vendor"] if m else "?"


def label_to_model_id(label):
    """
    从下拉标签解析 model id。三层兼容：
    1. 当前标签精确命中；
    2. 旧工作流里存的标签（价格已变动）→ 取第一个空格前的 id 段；
    3. 用户直接填了纯 model id → 原样返回（未收录也放行，交服务端判定）。
    """
    s = (label or "").strip()
    if not s:
        return None
    if s in _LABEL_TO_ID:
        return _LABEL_TO_ID[s]
    head = s.split(" ")[0].split("\t")[0]
    if head in _KNOWN_IDS:
        return head
    return head if head else None
