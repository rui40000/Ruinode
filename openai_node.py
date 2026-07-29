import numpy as np
import requests
import json
import base64
import io
import os
import re
from PIL import Image


def _clear_proxy_env():
    """
    清除可能导致 requests 连接错误的代理环境变量。
    仅在本模块加载时执行一次。
    """
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        os.environ.pop(key, None)

_clear_proxy_env()


def _sanitize_url(raw: str) -> str:
    """
    清理用户输入的 URL 片段：
    去除被 ComfyUI 前端残留的协议碎片、多余斜杠等，只保留 host/path 部分。
    """
    s = raw.strip()
    # 移除各种可能的协议残留:  "https:" / "http:" / "https://" / "http://"
    s = re.sub(r'^https?\s*:\s*/*/?\s*', '', s, flags=re.IGNORECASE)
    s = s.strip('/')
    return s


def _build_full_url(protocol: str, api_url: str) -> str:
    """
    用下拉框的协议名和文本框的地址拼出完整 URL。
    protocol 只会是 "https" 或 "http"（不含冒号和斜杠）。
    """
    host_path = _sanitize_url(api_url)
    if not host_path:
        host_path = 'api.openai.com/v1/chat/completions'
    # 唯一拼接 :// 的地方——纯 Python 字符串，不经过前端
    return protocol + '://' + host_path


def _build_proxy_url(raw_proxy: str) -> str:
    """
    用户可能输入 '127.0.0.1:7890' 或 'http://127.0.0.1:7890'，
    统一处理成带协议前缀的地址。
    """
    s = raw_proxy.strip()
    if not s:
        return ''
    # 已经有完整协议
    if re.match(r'^https?://', s, flags=re.IGNORECASE):
        return s
    # 移除残留碎片
    s = re.sub(r'^https?\s*:\s*/*/?\s*', '', s, flags=re.IGNORECASE)
    s = s.strip('/')
    if not s:
        return ''
    return 'http://' + s


class OpenAINode:
    """
    OpenAI API 连接节点
    ==================
    支持 OpenAI 及兼容协议的 API（DeepSeek、Moonshot、本地 Ollama 等）。
    - 纯文本模式：仅填写 user_prompt，进行对话生成
    - 多模态模式：连接 image_1~6，进行图像理解
    """

    # ────────── 输入定义 ──────────
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 下拉框只含纯英文单词，不含 :// ，杜绝被前端吞掉
                "protocol": (["https", "http"], {
                    "default": "https",
                    "tooltip": "接口协议。之所以单独做成下拉而不写进地址栏：\n"
                               "ComfyUI 前端会吞掉文本框里的 \"://\" 片段，\n"
                               "协议只能由后端拼接。"
                }),
                # 默认值不含任何协议前缀，杜绝被前端吞掉
                "api_url": ("STRING", {
                    "default": "api.openai.com/v1/chat/completions",
                    "multiline": False,
                    "tooltip": "接口地址，**不要带 http:// 或 https://**（协议见上方下拉）。\n"
                               "只填域名和路径，例如 api.openai.com/v1/chat/completions。\n"
                               "第三方中转填对应的域名即可。"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "API 密钥。\n"
                               "⚠ 工作流会连同此值一起保存，分享 json 前记得清空。"
                }),
                "model": ("STRING", {
                    "default": "gpt-4o",
                    "multiline": False,
                    "tooltip": "模型名，按服务商文档填写。\n"
                               "要传图就必须选支持视觉的型号，否则图会被忽略或直接报错。"
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                    "tooltip": "系统提示词：设定模型的角色与总体行为准则。\n"
                               "输出格式要求（如「只返回 JSON」）写在这里比写在用户\n"
                               "提示词里更稳定。"
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "用户提示词：这一次具体要模型做什么。\n"
                               "接了图像时，在这里描述针对图像的任务。"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子。多数服务商并不真正支持复现，\n"
                               "这里主要用于强制节点重新执行（改了它就不会走缓存）。"
                }),
            },
            "optional": {
                "image_1": ("IMAGE", {
                    "tooltip": "要一并发给模型的图像 1（需模型支持视觉）。\n"
                               "会按下方的最大边长压缩后转 base64 提交。"
                }),
                "image_2": ("IMAGE", {"tooltip": "图像 2。"}),
                "image_3": ("IMAGE", {"tooltip": "图像 3。"}),
                "image_4": ("IMAGE", {"tooltip": "图像 4。"}),
                "image_5": ("IMAGE", {"tooltip": "图像 5。"}),
                "image_6": ("IMAGE", {"tooltip": "图像 6。图越多越贵、越慢。"}),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "采样温度：越低越稳定保守，越高越发散。\n"
                               "要结构化/可解析的输出用 0~0.3；\n"
                               "要创意文案用 0.7~1.0。超过 1.2 常出现胡言乱语。"
                }),
                "max_tokens": ("INT", {
                    "default": 500,
                    "min": 1,
                    "max": 8192,
                    "tooltip": "回复的最大长度上限。\n"
                               "设小了会把回答从中间截断，长文任务记得调大。"
                }),
                "detail": (["low", "high", "auto"], {
                    "default": "auto",
                    "tooltip": "图像细节级别（OpenAI 视觉参数）：\n"
                               "low 便宜快速，只看大致内容；\n"
                               "high 会切块细看，认小字/细节更准但更贵；\n"
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
                # 代理地址也不带协议前缀，只填 IP:端口 即可
                "proxy_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HTTP 代理，**同样不要带协议前缀**，只填 IP:端口，\n"
                               "例如 127.0.0.1:7890。留空表示直连。"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "Rui-Node🐶/AI模型🤖"

    # ────────── 图像编码 ──────────
    @staticmethod
    def _encode_image(img_tensor, max_size):
        """将单张图像张量 [H,W,C] 编码为 base64 JPEG 字符串。"""
        img_np = img_tensor.cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        pil = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
        w, h = pil.size
        if max(w, h) > max_size:
            r = max_size / max(w, h)
            pil = pil.resize((max(1, int(w * r)), max(1, int(h * r))), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # ────────── 主函数 ──────────
    def generate_content(
        self,
        protocol,
        api_url,
        api_key,
        model,
        system_prompt,
        user_prompt,
        seed,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_6=None,
        temperature=0.3,
        max_tokens=500,
        detail="auto",
        image_max_size=1024,
        proxy_url="",
    ):
        # ---- 1. 拼接 URL（唯一产生 :// 的地方） ----
        full_url = _build_full_url(protocol, api_url)
        print(f"[Rui-Node] OpenAI -> {full_url}")

        # ---- 2. 收集图像 ----
        images = [
            img for img in (image_1, image_2, image_3, image_4, image_5, image_6)
            if img is not None
        ]

        # ---- 3. 构造 messages ----
        messages = [{"role": "system", "content": system_prompt}]

        if images:
            # ===== 多模态模式 =====
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
            # ===== 纯文本模式 =====
            text = (user_prompt or "").strip()
            if not text:
                return ("（错误：未提供图片也未提供提示词，请至少填写 user_prompt）",)
            messages.append({"role": "user", "content": text})

        # ---- 4. 请求 ----
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        proxies = None
        p = _build_proxy_url(proxy_url)
        if p:
            proxies = {"http": p, "https": p}

        try:
            resp = requests.post(full_url, headers=headers, json=payload,
                                 proxies=proxies, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and data["choices"]:
                return (data["choices"][0]["message"]["content"],)
            return (f"API 返回格式异常: {json.dumps(data, ensure_ascii=False)}",)
        except requests.exceptions.ConnectionError as e:
            return (f"连接失败（请检查 api_url 和网络）: {e}",)
        except requests.exceptions.Timeout:
            return ("请求超时（120s），请检查网络或 API 服务状态。",)
        except requests.exceptions.HTTPError as e:
            return (f"HTTP 错误 {resp.status_code}: {resp.text[:500]}",)
        except Exception as e:
            return (f"请求异常: {type(e).__name__}: {e}",)


# ────────── ComfyUI 注册 ──────────
NODE_CLASS_MAPPINGS = {
    "OpenAIAPINode": OpenAINode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIAPINode": "OpenAI API 连接 / OpenAI API Connector",
}
