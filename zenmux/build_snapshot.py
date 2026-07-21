# -*- coding: utf-8 -*-
"""
从 ZenMux 拉取模型列表，提炼成节点需要的精简快照。

运行：python build_snapshot.py
产出：models_snapshot.json（随节点分发，保证离线可用）

只保留文本类模型（输出模态含 text），并预先算好基础输入/输出价格，
避免节点运行时每次都解析 ZenMux 那套分段定价结构。
"""
import json
import os
import sys

import requests

# Windows GBK 控制台打印中文/特殊字符会崩，强制 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API = "https://zenmux.ai/api/v1/models"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "models_snapshot.json")


def base_price(pricings, key):
    """取某项价格的基础段（prompt_tokens 从 0 起的那一段），单位 USD / 百万 token。"""
    seg = (pricings or {}).get(key)
    if not seg:
        return None
    vals = [s for s in seg if isinstance(s, dict) and "value" in s]
    if not vals:
        return None
    base = None
    for s in vals:
        cond = s.get("conditions", {}).get("prompt_tokens", {})
        if cond.get("gte", 0) in (0, None):
            base = s
            break
    base = base or vals[0]
    try:
        return float(base["value"])
    except (TypeError, ValueError, KeyError):
        return None


def build(raw_models):
    out = []
    for m in raw_models:
        out_mod = m.get("output_modalities", [])
        if "text" not in out_mod:  # 只要文本类模型
            continue
        pr = m.get("pricings") or {}
        out.append({
            "id": m["id"],
            "display_name": m.get("display_name", m["id"]),
            # 厂商一律取模型 id 的前缀（"openai/gpt-5.4-nano" -> "openai"），
            # 与前端级联 JS 的过滤规则（label.split("/")[0]）保持同一真源。
            "vendor": m["id"].split("/")[0] if "/" in m["id"] else m.get("owned_by", "other"),
            "input_price": base_price(pr, "prompt"),
            "output_price": base_price(pr, "completion"),
            "context_length": m.get("context_length"),
            "input_modalities": m.get("input_modalities", []),
        })
    # 按厂商、再按 id 排序，方便前端级联展示
    out.sort(key=lambda x: (x["vendor"].lower(), x["id"].lower()))
    return out


def main():
    print(f"拉取 {API} ...")
    r = requests.get(API, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    models = build(data)
    payload = {
        "source": API,
        "count": len(models),
        "models": models,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"写入 {OUT}：{len(models)} 个文本模型")
    vendors = sorted(set(m["vendor"] for m in models))
    print(f"厂商 {len(vendors)} 个：{', '.join(vendors)}")
    assert any(m["id"] == "openai/gpt-5.4-nano" for m in models), "默认模型缺失！"
    print("默认模型 openai/gpt-5.4-nano 存在 ✓")


if __name__ == "__main__":
    sys.exit(main())
