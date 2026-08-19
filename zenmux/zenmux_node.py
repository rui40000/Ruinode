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
import random
import re
import time

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


# ── 网络层自动重连 ──
# 可重试的 HTTP 状态：限流与服务端临时故障，换个时间再来往往就好了。
# 4xx（除 408/429）是参数或鉴权问题，重试纯属浪费，不列入。
_RETRIABLE_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}

# 这几类异常的共同点是「没拿到完整响应」，重连通常能恢复：
#   ConnectionError —— 含 SSLError/SSLEOFError（握手被打断）、连接被重置
#   Timeout         —— 连接超时与读超时
#   ChunkedEncodingError / JSONDecodeError —— 响应体传到一半断了
_RETRIABLE_EXC = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    json.JSONDecodeError,
)


def _retry_wait(attempt: int, resp=None) -> float:
    """指数退避 + 随机抖动，返回本次该等几秒。attempt 从 0 起。

    抖动不是可有可无的装饰：一条工作流里常有多个 API 节点同时失败，
    没有抖动它们会在同一毫秒一起重连，把刚缓过来的服务端再打垮一次。
    服务端明确给了 Retry-After 时以它为准。
    """
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return max(0.5, min(float(ra), 60.0))
            except (TypeError, ValueError):
                pass
    base = min(2.0 ** attempt, 16.0)          # 1s → 2s → 4s → 8s → 16s 封顶
    return round(base * random.uniform(0.75, 1.25), 2)


def _build_proxy_url(raw_proxy: str) -> str:
    """把 '127.0.0.1:7890' 或 'http://127.0.0.1:7890' 统一成带协议的地址。"""
    s = (raw_proxy or "").strip()
    if not s:
        return ''
    if re.match(r'^https?://', s, flags=re.IGNORECASE):
        return s
    s = re.sub(r'^https?\s*:\s*/*/?\s*', '', s, flags=re.IGNORECASE).strip('/')
    return ('http://' + s) if s else ''


# 遇到「参数不被支持 / 已弃用」的 400 时，可安全剔除的采样参数
# （绝不触碰 model / messages 等核心字段）
_ADJUSTABLE_PARAMS = ("temperature", "top_p", "seed", "max_tokens",
                      "max_completion_tokens", "frequency_penalty",
                      "presence_penalty", "top_k")
# 判定「这条 400 是参数问题」的关键词（命中才尝试剔除重试）
_PARAM_ERR_HINTS = ("deprecat", "unsupport", "not support", "not allowed",
                    "invalid", "unexpected", "unknown", "must be", "cannot",
                    "not permitted", "removed")


def _diagnose_param(resp, payload):
    """
    从 400 响应里判断是哪个采样参数导致失败，返回处理指令：
      ("drop", 参数名)     —— 从 payload 剔除该参数后重试
      ("rename_mct", ...)  —— 把 max_tokens 改名为 max_completion_tokens 后重试
      None                 —— 非参数类错误，不重试
    ZenMux 各模型参数规则不一（claude-sonnet-5 弃用 temperature、gpt-5 reasoning
    系要求 max_completion_tokens 等），靠错误消息动态识别，免维护静态清单。
    """
    try:
        msg = (resp.json().get("error", {}) or {}).get("message", "") or resp.text or ""
    except Exception:
        msg = getattr(resp, "text", "") or ""
    low = msg.lower()
    # 特例：OpenAI reasoning 系要求用 max_completion_tokens 替代 max_tokens
    if "max_completion_tokens" in low and "max_tokens" in payload:
        return ("rename_mct", "max_tokens")
    # 反引号/引号里的参数名优先（错误消息通常形如  `temperature` is deprecated）
    for name in re.findall(r"""[`'"]([a-zA-Z_]+)[`'"]""", msg):
        if name in payload and name in _ADJUSTABLE_PARAMS:
            return ("drop", name)
    # 回退：错误像参数问题时，扫描 payload 里哪个参数名出现在消息中
    if any(h in low for h in _PARAM_ERR_HINTS):
        for name in _ADJUSTABLE_PARAMS:
            if name in payload and name in low:
                return ("drop", name)
    return None


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
                    "tooltip": "ZenMux 的 API Key。\n"
                               "⚠ 工作流会连同此值一起保存，分享 json 前记得清空。"
                }),
                # 全量模型标签（含价格），已按厂商排序聚类；
                # 下拉搜索框输入厂商前缀（如 "qwen/"）即可过滤。
                "model": (model_labels, {
                    "default": default_label,
                    "tooltip": "模型，标签里直接带了输入/输出单价。\n"
                               "列表按厂商聚类排序——在下拉的搜索框输入厂商前缀\n"
                               "（如 qwen/ 、anthropic/ ）即可快速过滤。\n"
                               "要传图请选支持视觉的型号，否则图会被忽略。"
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                    "tooltip": "系统提示词：设定模型的角色与总体行为准则。\n"
                               "输出格式要求（如「只返回 JSON」）写在这里最稳定。"
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "用户提示词：这一次具体要模型做什么。"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子。多数模型并不真正支持复现，\n"
                               "这里主要用于强制节点重新执行（改了它就不会走缓存）。"
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "采样温度：越低越稳定保守，越高越发散。\n"
                               "结构化输出用 0~0.3，创意文案用 0.7~1.0。\n"
                               "部分新模型已弃用该参数，节点会自动重试并剔除它。"
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "核采样：只在累计概率前 top_p 的词里挑。\n"
                               "与温度作用重叠，一般固定 1.0 只调温度，别两个一起动。"
                }),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 1,
                    "max": 200000,
                    "tooltip": "回复的最大长度上限。设小了会把回答从中间截断。\n"
                               "注意它同时是费用上限的重要因素。"
                }),
                # 多模态图像输入（模型需支持 image 输入才有意义）
                "image_1": ("IMAGE", {
                    "tooltip": "要一并发给模型的图像 1（需模型支持视觉）。\n"
                               "会按下方最大边长压缩后转 base64 提交。"
                }),
                "image_2": ("IMAGE", {"tooltip": "图像 2。"}),
                "image_3": ("IMAGE", {"tooltip": "图像 3。"}),
                "image_4": ("IMAGE", {"tooltip": "图像 4。"}),
                "image_5": ("IMAGE", {"tooltip": "图像 5。"}),
                "image_6": ("IMAGE", {"tooltip": "图像 6。图越多越贵、越慢。"}),
                "detail": (["auto", "low", "high"], {
                    "default": "auto",
                    "tooltip": "图像细节级别：\n"
                               "low 便宜快速，只看大致内容；\n"
                               "high 切块细看，认小字/细节更准但更贵；\n"
                               "auto 由服务端决定。"
                }),
                "image_max_size": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "上传前把图缩放到的最大边长。\n"
                               "调小可显著省钱提速，但小字与细节会看不清。"
                }),
                # 高级：一般无需改动，留空即用官方地址（无需写 https://，会自动补全）
                "base_url": ("STRING", {
                    "default": DEFAULT_BASE_URL_INPUT,
                    "multiline": False,
                    "tooltip": "接口地址，一般不用改。\n"
                               "**不要写 https://** —— ComfyUI 前端会吞掉 \"://\"，\n"
                               "协议由后端自动补全，这里只填域名和路径。"
                }),
                "proxy_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HTTP 代理，同样不要带协议前缀，只填 IP:端口，\n"
                               "例如 127.0.0.1:7890。留空表示直连。"
                }),
                # usage_stats 里人民币换算用的汇率，可按当日牌价自行调整
                "max_retries": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "网络失败后的自动重连次数，0 表示不重连。\n\n"
                               "会触发重连的情况：SSL 握手被打断\n"
                               "（SSLEOFError）、连接被重置、读超时、响应体\n"
                               "截断，以及 429 限流和 5xx 服务端临时故障。\n\n"
                               "不会重连的情况：参数类 400、鉴权类 401/403 ——\n"
                               "这些重试多少次都是同样的结果。\n\n"
                               "退避按 1s→2s→4s 指数增长并带随机抖动，避免\n"
                               "多个节点同时重连再次压垮服务端；服务端给了\n"
                               "Retry-After 时以它为准。"
                }),
                "timeout": ("INT", {
                    "default": 180,
                    "min": 10,
                    "max": 1800,
                    "step": 10,
                    "tooltip": "单次请求的超时秒数。\n"
                               "长文本或多图推理较慢时可调大。\n"
                               "超时会计入上面的重连次数。"
                }),
                "usd_to_cny": ("FLOAT", {
                    "default": 7.2,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "美元兑人民币汇率，仅用于把 usage_stats 输出里的\n"
                               "费用换算成人民币显示，不影响实际计费。\n"
                               "可按当日牌价自行调整。"
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
        max_retries=3,
        timeout=180,
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
        dropped = []  # 记录被剔除/改名的参数，供最终报错时提示
        last_net_err = ""
        total_tries = max(1, int(max_retries) + 1)

        for attempt in range(total_tries):
            try:
                # 自适应参数重试：ZenMux 聚合的部分模型弃用/不支持某些采样参数
                # （claude-sonnet-5 弃用 temperature、gpt-5 reasoning 系要 max_completion_tokens
                #  等），命中即剔除/改名后重试，正常请求不受影响、零额外开销。
                # 这层不消耗网络重连次数——两者是性质不同的问题。
                for _ in range(len(_ADJUSTABLE_PARAMS) + 2):
                    resp = requests.post(chat_url, headers=headers, json=payload,
                                         proxies=proxies, timeout=timeout)
                    if resp.status_code == 400:
                        fix = _diagnose_param(resp, payload)
                        if fix:
                            action, param = fix
                            if action == "rename_mct":
                                payload["max_completion_tokens"] = payload.pop("max_tokens")
                                dropped.append("max_tokens→max_completion_tokens")
                                print("[Rui-Node] ZenMux: 该模型要求 max_completion_tokens，"
                                      "已改名重试")
                            else:
                                payload.pop(param, None)
                                dropped.append(param)
                                print(f"[Rui-Node] ZenMux: 该模型不支持参数 '{param}'，"
                                      "已剔除后重试")
                            continue
                    break

                # 限流 / 服务端临时故障：计入网络重连
                if resp.status_code in _RETRIABLE_STATUS and attempt < total_tries - 1:
                    wait = _retry_wait(attempt, resp)
                    print(f"[Rui-Node] ZenMux: 服务端返回 {resp.status_code}，"
                          f"{wait:.1f}s 后重试"
                          f"（第 {attempt + 1}/{total_tries - 1} 次重连）")
                    time.sleep(wait)
                    continue

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
                    if attempt:
                        print(f"[Rui-Node] ZenMux: 第 {attempt} 次重连后成功")
                    return (content or "", model_id, stats)
                # 格式异常：无正文可计字数，但 usage 仍尽量取真实值
                stats = self._build_usage_stats(model_id, data.get("usage"), usd_to_cny)
                return (f"API 返回格式异常: {json.dumps(data, ensure_ascii=False)[:800]}",
                        model_id, stats)

            except _RETRIABLE_EXC as e:
                last_net_err = f"{type(e).__name__}: {e}"
                if attempt < total_tries - 1:
                    wait = _retry_wait(attempt)
                    print(f"[Rui-Node] ZenMux: 连接失败（{type(e).__name__}），"
                          f"{wait:.1f}s 后重连"
                          f"（第 {attempt + 1}/{total_tries - 1} 次）")
                    time.sleep(wait)
                    continue
                tail = (f"（已自动重连 {max_retries} 次仍失败；"
                        f"可调大 max_retries，或检查网络/代理）"
                        if max_retries else "（max_retries=0，未重连）")
                return (f"连接失败（请检查网络/代理）: {last_net_err}{tail}",
                        model_id, zero_stats)

            except requests.exceptions.HTTPError:
                # 参数与鉴权类错误重试无意义，直接报错
                code = resp.status_code if resp is not None else "?"
                body = resp.text[:600] if resp is not None else ""
                hint = f"（已尝试剔除参数 {', '.join(dropped)} 仍失败）" if (code == 400 and dropped) else ""
                if attempt:      # 重连过仍是这个状态，明说，省得用户以为没重试
                    hint += f"（已自动重连 {attempt} 次仍返回该状态）"
                return (f"HTTP 错误 {code}: {body}{hint}", model_id, zero_stats)

            except Exception as e:
                return (f"请求异常: {type(e).__name__}: {e}", model_id, zero_stats)

        # 循环内必定 return；这条仅作兜底，避免异常路径返回 None
        return ("请求未能完成，请重试或检查服务状态。", model_id, zero_stats)


# ────────── ComfyUI 注册 ──────────
NODE_CLASS_MAPPINGS = {
    "ZenMuxAPINode": ZenMuxNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZenMuxAPINode": "ZenMux API 连接 / ZenMux API Connector",
}
