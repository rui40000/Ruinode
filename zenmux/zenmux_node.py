# -*- coding: utf-8 -*-
"""
ZenMux API 连接节点
===================
通过 ZenMux 聚合平台（https://zenmux.ai）调用其收录的所有文本类模型。

特性：
- 模型下拉覆盖 ZenMux 全部文本模型，按「厂商/模型名」排序聚类；
  ComfyUI 下拉自带搜索，输入厂商前缀（如 "anthropic/"）即可快速过滤。
- 每个模型选项后面直接标注输入/输出价格（USD / 百万 token）。
- usage_stats 输出单次运行的 token 消耗与费用（按快照单价折算，
  汇率可用 usd_to_cny 参数调整）。
- 具备常规 API 节点的完整参数：api_key、system/user prompt、seed、
  temperature、top_p、max_tokens、以及可选的多模态图像输入与代理。
- 默认模型 openai/gpt-5.4-nano。
- 随节点分发 models_snapshot.json，无网络也能列出模型；价格与列表可用
  build_snapshot.py 重新拉取更新。

注：ZenMux 采用 OpenAI 兼容协议，chat 端点为
    https://zenmux.ai/api/v1/chat/completions
"""
import base64
import io
import json
import os
import re

import numpy as np
import requests
from PIL import Image

from .model_registry import (
    DEFAULT_MODEL_ID,
    default_model_label,
    all_model_labels,
    label_to_model_id,
    model_label_by_id,
    model_prices,
)

# ZenMux 平台固定地址（OpenAI 兼容）
DEFAULT_BASE_URL = "https://zenmux.ai/api/v1"
# 输入框默认值不带 "://"——本仓库实测 ComfyUI 前端会吞掉文本框里的
# 协议片段（见 openai_node.py 的同款处理），后端 _build_chat_url 会自动补 https。
DEFAULT_BASE_URL_INPUT = "zenmux.ai/api/v1"


def _clear_proxy_env():
    """清除可能干扰 requests 的代理环境变量（仅本模块加载时执行一次）。"""
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ.pop(key, None)


_clear_proxy_env()


def _build_chat_url(base_url: str) -> str:
    """由 base_url 拼出 chat/completions 端点，容忍用户填了/没填结尾斜杠。"""
    s = (base_url or "").strip()
    if not s:
        s = DEFAULT_BASE_URL
    # 补协议
    if not re.match(r'^https?://', s, flags=re.IGNORECASE):
        s = 'https://' + s.lstrip('/')
    s = s.rstrip('/')
    # 用户可能已经把 /chat/completions 填进去了
    if s.lower().endswith('/chat/completions'):
        return s
    return s + '/chat/completions'


def _build_proxy_url(raw_proxy: str) -> str:
    """把 '127.0.0.1:7890' 或 'http://127.0.0.1:7890' 统一成带协议的地址。"""
    s = (raw_proxy or "").strip()
    if not s:
        return ''
    if re.match(r'^https?://', s, flags=re.IGNORECASE):
        return s
    s = re.sub(r'^https?\s*:\s*/*/?\s*', '', s, flags=re.IGNORECASE).strip('/')
    return ('http://' + s) if s else ''


class ZenMuxNode:
    """ZenMux API 连接节点。"""

    # ────────── 输入定义 ──────────
    @classmethod
    def INPUT_TYPES(cls):
        model_labels = all_model_labels()
        default_label = default_model_label()
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                # 全量模型标签（含价格），已按厂商排序聚类；
                # 下拉搜索框输入厂商前缀（如 "qwen/"）即可过滤。
                "model": (model_labels, {
                    "default": default_label,
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 200000,
                }),
                # 多模态图像输入（模型需支持 image 输入才有意义）
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "detail": (["auto", "low", "high"], {
                    "default": "auto",
                }),
                "image_max_size": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                }),
                # 高级：一般无需改动，留空即用官方地址（无需写 https://，会自动补全）
                "base_url": ("STRING", {
                    "default": DEFAULT_BASE_URL_INPUT,
                    "multiline": False,
                }),
                "proxy_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                # usage_stats 里人民币换算用的汇率，可按当日牌价自行调整
                "usd_to_cny": ("FLOAT", {
                    "default": 7.2,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "model_id", "usage_stats")
    FUNCTION = "generate"
    CATEGORY = "Rui-Node🐶/AI模型🤖"

    # ────────── 宽松校验 ──────────
    @classmethod
    def VALIDATE_INPUTS(cls, model):
        """
        接管 model 下拉的校验，替代 ComfyUI 内置的「值必须在候选列表里」
        检查：价格快照更新后，旧工作流里保存的标签（带旧价格）不再逐字
        匹配新列表，但只要能解析出 model id 就应放行，避免整个工作流
        被判为无效。
        """
        if label_to_model_id(model) is None:
            return f"无法从 '{model}' 解析出 ZenMux 模型 id"
        return True

    # ────────── 图像编码 ──────────
    @staticmethod
    def _encode_image(img_tensor, max_size):
        """把单张图像张量 [H,W,C]（0~1）编码为 base64 JPEG。"""
        img_np = np.clip(img_tensor.cpu().numpy(), 0, 1)
        pil = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
        w, h = pil.size
        if max(w, h) > max_size:
            r = max_size / max(w, h)
            pil = pil.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # ────────── 消耗统计 ──────────
    @staticmethod
    def _fmt_cost(v):
        """费用格式化：最多 6 位小数并去尾零；未知为 '?'。"""
        if v is None:
            return "?"
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"

    @staticmethod
    def _build_usage_stats(model_id, usage, usd_to_cny, out_text=None):
        """
        由 API 响应的 usage、输出文本与快照单价生成四行消耗统计：
            token消耗，输入：XXX，输出：XXX
            输出文字数量：XXX
            模型类型：openai/gpt-5.4-nano [入$0.2/M 出$1.25/M]
            价格换算，美元：XXX，人民币：XXX
        usage 缺失/单价未知的项以 '?' 呈现；请求未发生时传 usage=None 记为 0 消耗。
        out_text 为模型返回的文本，字数按字符数计（含标点）；无输出记 0。
        """
        if not isinstance(usage, dict):
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        in_price, out_price = model_prices(model_id)

        usd = None
        if (isinstance(in_tok, (int, float)) and isinstance(out_tok, (int, float))
                and in_price is not None and out_price is not None):
            usd = in_tok / 1e6 * in_price + out_tok / 1e6 * out_price
        cny = usd * usd_to_cny if usd is not None else None

        n_chars = len(out_text) if isinstance(out_text, str) else 0
        tok = lambda t: str(int(t)) if isinstance(t, (int, float)) else "?"  # noqa: E731
        return (f"token消耗，输入：{tok(in_tok)}，输出：{tok(out_tok)}\n"
                f"输出文字数量：{n_chars}\n"
                f"模型类型：{model_label_by_id(model_id)}\n"
                f"价格换算，美元：{ZenMuxNode._fmt_cost(usd)}，人民币：{ZenMuxNode._fmt_cost(cny)}")

    # ────────── 主函数 ──────────
    def generate(
        self,
        api_key,
        model,
        system_prompt,
        user_prompt,
        seed,
        temperature=0.7,
        top_p=1.0,
        max_tokens=1024,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        detail="auto",
        image_max_size=1024,
        base_url=DEFAULT_BASE_URL,
        proxy_url="",
        usd_to_cny=7.2,
    ):
        # ---- 1. 从下拉标签解析真实 model id ----
        model_id = label_to_model_id(model) or DEFAULT_MODEL_ID
        chat_url = _build_chat_url(base_url)
        print(f"[Rui-Node] ZenMux -> {chat_url}  model={model_id}")

        # 请求未成功前的兜底统计（0 消耗）
        zero_stats = self._build_usage_stats(model_id, None, usd_to_cny)

        if not (api_key or "").strip():
            return ("（错误：未填写 api_key，请在节点里填入 ZenMux 的 API Key）",
                    model_id, zero_stats)

        # ---- 2. 收集图像 ----
        images = [
            img for img in (image_1, image_2, image_3, image_4, image_5, image_6)
            if img is not None
        ]

        # ---- 3. 构造 messages ----
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        if images:
            parts = []
            if user_prompt and user_prompt.strip():
                parts.append({"type": "text", "text": user_prompt})
            for img in images:
                b64 = self._encode_image(img[0], image_max_size)
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": detail,
                    },
                })
            if not parts:
                parts.append({"type": "text", "text": " "})
            messages.append({"role": "user", "content": parts})
        else:
            text = (user_prompt or "").strip()
            if not text:
                return ("（错误：未提供图片也未提供提示词，请至少填写 user_prompt）",
                        model_id, zero_stats)
            messages.append({"role": "user", "content": text})

        # ---- 4. 请求 ----
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key.strip()}",
        }
        payload = {
            "model": model_id,
            "messages": messages,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        proxies = None
        p = _build_proxy_url(proxy_url)
        if p:
            proxies = {"http": p, "https": p}

        resp = None
        try:
            resp = requests.post(chat_url, headers=headers, json=payload,
                                 proxies=proxies, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            if data.get("choices"):
                msg = data["choices"][0].get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):  # 少数模型返回分段内容
                    content = "".join(
                        seg.get("text", "") for seg in content if isinstance(seg, dict)
                    )
                stats = self._build_usage_stats(model_id, data.get("usage"),
                                                usd_to_cny, content or "")
                return (content or "", model_id, stats)
            # 格式异常：无正文可计字数，但 usage 仍尽量取真实值
            stats = self._build_usage_stats(model_id, data.get("usage"), usd_to_cny)
            return (f"API 返回格式异常: {json.dumps(data, ensure_ascii=False)[:800]}",
                    model_id, stats)
        except requests.exceptions.ConnectionError as e:
            return (f"连接失败（请检查网络/代理）: {e}", model_id, zero_stats)
        except requests.exceptions.Timeout:
            return ("请求超时（180s），请检查网络或 ZenMux 服务状态。", model_id, zero_stats)
        except requests.exceptions.HTTPError:
            code = resp.status_code if resp is not None else "?"
            body = resp.text[:600] if resp is not None else ""
            return (f"HTTP 错误 {code}: {body}", model_id, zero_stats)
        except Exception as e:
            return (f"请求异常: {type(e).__name__}: {e}", model_id, zero_stats)


# ────────── ComfyUI 注册 ──────────
NODE_CLASS_MAPPINGS = {
    "ZenMuxAPINode": ZenMuxNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZenMuxAPINode": "ZenMux API 连接 / ZenMux API Connector",
}
