# -*- coding: utf-8 -*-
"""
越光 API 连接节点
=================
通过越光（Nebula）聚合平台调用其收录的文本类模型。
接口为 OpenAI 兼容协议，chat 端点：
    https://llm.ai-nebula.com/v1/chat/completions

与 ZenMux 节点的差异：
- 越光的可用模型与价格由官方规范文档给定，没有可枚举的模型接口，
  因此模型清单内置在 model_registry.py 里，不做在线快照；
- 越光的 model id **不带厂商前缀**（gpt-4o 而非 openai/gpt-4o），
  下拉里同厂商靠排序聚在一起，搜索时输 gpt / claude / deepseek / kimi 过滤。

其余行为与 ZenMux 节点一致，包括那套「自适应参数重试」——
部分模型弃用 temperature、或要求用 max_completion_tokens 取代 max_tokens，
命中这类 400 报错时会剔除/改名对应参数后自动重试，正常请求零额外开销。
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
    all_model_labels,
    default_model_label,
    label_to_model_id,
    model_label_by_id,
    model_prices,
    model_vendor,
)

# 越光平台固定地址（OpenAI 兼容）
DEFAULT_BASE_URL = "https://llm.ai-nebula.com/v1"
# 输入框默认值不带 "://"——ComfyUI 前端会吞掉文本框里的协议片段
# （本仓库为此修过多次），协议由 _build_chat_url 自动补全。
DEFAULT_BASE_URL_INPUT = "llm.ai-nebula.com/v1"


def _clear_proxy_env():
    """清除可能干扰 requests 的代理环境变量（仅本模块加载时执行一次）。"""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)


_clear_proxy_env()


def _build_chat_url(base_url: str) -> str:
    """由 base_url 拼出 chat/completions 端点，容忍结尾斜杠有无。"""
    s = (base_url or "").strip()
    if not s:
        s = DEFAULT_BASE_URL
    if not re.match(r"^https?://", s, flags=re.IGNORECASE):
        s = "https://" + s.lstrip("/")
    s = s.rstrip("/")
    if s.lower().endswith("/chat/completions"):
        return s
    return s + "/chat/completions"


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
        return ""
    if re.match(r"^https?://", s, flags=re.IGNORECASE):
        return s
    s = re.sub(r"^https?\s*:\s*/*/?\s*", "", s, flags=re.IGNORECASE).strip("/")
    return ("http://" + s) if s else ""


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
    各模型的参数规则不一（新版 claude 弃用 temperature、OpenAI reasoning 系
    要求 max_completion_tokens 等），靠错误消息动态识别，免维护静态清单。
    """
    try:
        msg = (resp.json().get("error", {}) or {}).get("message", "") or resp.text or ""
    except Exception:
        msg = getattr(resp, "text", "") or ""
    low = msg.lower()
    if "max_completion_tokens" in low and "max_tokens" in payload:
        return ("rename_mct", "max_tokens")
    for name in re.findall(r"""[`'"]([a-zA-Z_]+)[`'"]""", msg):
        if name in payload and name in _ADJUSTABLE_PARAMS:
            return ("drop", name)
    if any(h in low for h in _PARAM_ERR_HINTS):
        for name in _ADJUSTABLE_PARAMS:
            if name in payload and name in low:
                return ("drop", name)
    return None


class YueGuangNode:
    """越光 API 连接节点。"""

    @classmethod
    def INPUT_TYPES(cls):
        model_labels = all_model_labels()
        default_label = default_model_label()
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "越光的 API Key（形如 sk-xxxxxxxx）。\n"
                               "⚠ 工作流会连同此值一起保存，分享 json 前记得清空。"
                }),
                "model": (model_labels, {
                    "default": default_label,
                    "tooltip": "模型，标签里直接带了输入/输出单价（USD/百万 token）。\n"
                               "越光的 model id 不带厂商前缀，下拉里同厂商是靠排序\n"
                               "聚在一起的 —— 在下拉的搜索框输 gpt / claude /\n"
                               "deepseek / kimi 即可快速过滤。\n"
                               "默认 deepseek-v4-flash 是表里最便宜的（$0.1/$0.3）。"
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
                               "部分新模型已弃用该参数，节点会自动剔除后重试。"
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
                "image_1": ("IMAGE", {
                    "tooltip": "要一并发给模型的图像 1（需所选模型支持视觉）。\n"
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

    @classmethod
    def VALIDATE_INPUTS(cls, model):
        """
        接管 model 下拉的校验：价格表更新后，旧工作流里保存的标签
        （带旧价格）不再逐字匹配新列表，但只要能解析出 model id 就应放行，
        避免整个工作流被判为无效。
        """
        if label_to_model_id(model) is None:
            return f"无法从 '{model}' 解析出越光模型 id"
        return True

    @staticmethod
    def _encode_image(img_tensor, max_size):
        """把单张图像张量 [H,W,C]（0~1）编码为 base64 JPEG。"""
        img_np = np.clip(img_tensor.cpu().numpy(), 0, 1)
        if img_np.shape[-1] == 4:          # 带 alpha 的输入丢掉 alpha
            img_np = img_np[..., :3]
        pil = Image.fromarray((img_np * 255).astype(np.uint8), "RGB")
        w, h = pil.size
        if max(w, h) > max_size:
            r = max_size / max(w, h)
            pil = pil.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

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
        由 API 响应的 usage、输出文本与内置单价生成五行消耗统计：
            token消耗，输入：XXX，输出：XXX
            输出文字数量：XXX
            厂商：OpenAI
            模型类型：gpt-4o [入$2.5/M 出$10/M]
            价格换算，美元：XXX，人民币：XXX
        usage 缺失/单价未知的项以 '?' 呈现；请求未发生时传 usage=None 记为 0 消耗。
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
                f"厂商：{model_vendor(model_id)}\n"
                f"模型类型：{model_label_by_id(model_id)}\n"
                f"价格换算，美元：{YueGuangNode._fmt_cost(usd)}，"
                f"人民币：{YueGuangNode._fmt_cost(cny)}")

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
        base_url=DEFAULT_BASE_URL_INPUT,
        proxy_url="",
        usd_to_cny=7.2,
        max_retries=3,
        timeout=180,
    ):
        model_id = label_to_model_id(model) or DEFAULT_MODEL_ID
        chat_url = _build_chat_url(base_url)
        print(f"[Rui-Node] 越光 -> {chat_url}  model={model_id}")

        zero_stats = self._build_usage_stats(model_id, None, usd_to_cny)

        if not (api_key or "").strip():
            return ("（错误：未填写 api_key，请在节点里填入越光的 API Key）",
                    model_id, zero_stats)

        images = [img for img in (image_1, image_2, image_3, image_4,
                                  image_5, image_6) if img is not None]

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
        dropped = []
        last_net_err = ""
        total_tries = max(1, int(max_retries) + 1)

        for attempt in range(total_tries):
            try:
                # 自适应参数重试：部分模型弃用/不支持某些采样参数，
                # 命中即剔除或改名后重试；正常请求不受影响、零额外开销。
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
                                print("[Rui-Node] 越光: 该模型要求 max_completion_tokens，"
                                      "已改名重试")
                            else:
                                payload.pop(param, None)
                                dropped.append(param)
                                print(f"[Rui-Node] 越光: 该模型不支持参数 '{param}'，"
                                      "已剔除后重试")
                            continue
                    break

                # 限流 / 服务端临时故障：计入网络重连
                if resp.status_code in _RETRIABLE_STATUS and attempt < total_tries - 1:
                    wait = _retry_wait(attempt, resp)
                    print(f"[Rui-Node] 越光: 服务端返回 {resp.status_code}，"
                          f"{wait:.1f}s 后重试"
                          f"（第 {attempt + 1}/{total_tries - 1} 次重连）")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                if data.get("choices"):
                    msg = data["choices"][0].get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, list):      # 少数模型返回分段内容
                        content = "".join(
                            seg.get("text", "") for seg in content
                            if isinstance(seg, dict)
                        )
                    stats = self._build_usage_stats(model_id, data.get("usage"),
                                                    usd_to_cny, content or "")
                    if attempt:
                        print(f"[Rui-Node] 越光: 第 {attempt} 次重连后成功")
                    return (content or "", model_id, stats)
                stats = self._build_usage_stats(model_id, data.get("usage"), usd_to_cny)
                return (f"API 返回格式异常: {json.dumps(data, ensure_ascii=False)[:800]}",
                        model_id, stats)

            except _RETRIABLE_EXC as e:
                last_net_err = f"{type(e).__name__}: {e}"
                if attempt < total_tries - 1:
                    wait = _retry_wait(attempt)
                    print(f"[Rui-Node] 越光: 连接失败（{type(e).__name__}），"
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
                hint = (f"（已尝试剔除参数 {', '.join(dropped)} 仍失败）"
                        if (code == 400 and dropped) else "")
                if attempt:      # 重连过仍是这个状态，明说，省得用户以为没重试
                    hint += f"（已自动重连 {attempt} 次仍返回该状态）"
                return (f"HTTP 错误 {code}: {body}{hint}", model_id, zero_stats)

            except Exception as e:
                return (f"请求异常: {type(e).__name__}: {e}", model_id, zero_stats)

        # 循环内必定 return；这条仅作兜底，避免异常路径返回 None
        return ("请求未能完成，请重试或检查服务状态。", model_id, zero_stats)


NODE_CLASS_MAPPINGS = {
    "YueGuangAPINode": YueGuangNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "YueGuangAPINode": "越光 API 连接 / YueGuang API Connector",
}
