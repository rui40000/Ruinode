import numpy as np
import requests
import json
import base64
import io
import os
import re
from PIL import Image

# 快速解决方案：清除可能导致连接错误的代理环境变量
# Fast solution: Clear proxy environment variables that might cause connection errors
# 许多用户在使用 requests 库连接 OpenAI API 时会遇到 ProxyError
# 这是因为 Python 环境可能读取了不正确的系统代理设置
# Many users encounter ProxyError when connecting to OpenAI API with requests
# This is because the Python environment might read incorrect system proxy settings
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''


def _build_url(protocol: str, api_url: str) -> str:
    """
    将协议和地址拼接为完整 URL。
    兼容多种用户输入格式：
      - 'api.openai.com/v1/chat/completions'         （标准，无协议前缀）
      - 'https://api.openai.com/v1/chat/completions'  （用户手动写了完整 URL）
      - 'https:api.openai.com/...'                     （被 ComfyUI 吞掉了 //）
      - 'http://127.0.0.1:5000/v1/chat/completions'   （本地服务）
    """
    url = api_url.strip()

    # 如果用户手动输入了完整 URL（含协议），直接修复并使用
    if re.match(r'^https?://.+', url):
        return url

    # 去掉可能残留的被截断的协议前缀，如 "https:" "http:" "https://" "http://"
    url = re.sub(r'^https?:/*', '', url)

    # 去掉首尾多余的斜杠
    url = url.strip('/')

    if not url:
        # 全部被吞了，回退到 OpenAI 默认地址
        url = 'api.openai.com/v1/chat/completions'

    return f"{protocol}{url}"


class OpenAINode:
    """
    OpenAI API 节点：
    支持连接 OpenAI 及其兼容 API（如 DeepSeek, Moonshot 等），
    支持文本生成和多模态图像理解。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "protocol": (["https://", "http://"], {
                    "default": "https://"
                }),
                "api_url": ("STRING", {
                    "default": "api.openai.com/v1/chat/completions",
                    "multiline": False,
                    "placeholder": "api.openai.com/v1/chat/completions"
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "model": ("STRING", {
                    "default": "gpt-4o",
                    "multiline": False
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True
                }),
                "user_prompt": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "max_tokens": ("INT", {
                    "default": 500,
                    "min": 1,
                    "max": 8192
                }),
                "detail": (["low", "high", "auto"], {
                    "default": "auto"
                }),
                "image_max_size": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "proxy_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g., http://127.0.0.1:7890"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"
    CATEGORY = "Rui-Node🐶/AI模型🤖"

    def _encode_image_tensor(self, img_tensor, image_max_size):
        img_np = img_tensor.cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
        width, height = img_pil.size
        if max(width, height) > image_max_size:
            ratio = image_max_size / max(width, height)
            img_pil = img_pil.resize(
                (max(1, int(width * ratio)), max(1, int(height * ratio))),
                Image.LANCZOS
            )
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

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
        proxy_url=""
    ):
        """
        调用 OpenAI API 生成内容。
        - 有图像时：使用多模态格式（image_url + text）发送请求
        - 无图像时：使用纯文本格式发送请求，实现普通对话
        """

        # ---- 拼接完整 URL ----
        full_url = _build_url(protocol, api_url)
        print(f"[Rui-Node🐶 OpenAI] 请求地址: {full_url}")

        all_images = [
            img for img in [
                image_1,
                image_2,
                image_3,
                image_4,
                image_5,
                image_6,
            ] if img is not None
        ]

        # ---- 构造 messages ----
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if all_images:
            # 多模态模式：图像 + 文本
            user_content = []

            if user_prompt:
                user_content.append({
                    "type": "text",
                    "text": user_prompt
                })

            for image in all_images:
                img_tensor = image[0]
                img_base64 = self._encode_image_tensor(img_tensor, image_max_size)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": detail
                    }
                })

            if not user_content:
                user_content.append({
                    "type": "text",
                    "text": " "
                })

            messages.append({
                "role": "user",
                "content": user_content
            })
        else:
            # 纯文本模式：无图像，直接发送文本对话
            prompt_text = user_prompt.strip() if user_prompt else ""
            if not prompt_text:
                return ("（错误：未提供图片也未提供提示词，请至少输入 user_prompt）",)

            messages.append({
                "role": "user",
                "content": prompt_text
            })

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": model,
            "messages": messages,
            "seed": seed,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        proxies = None
        if proxy_url and proxy_url.strip():
            proxies = {
                "http": proxy_url,
                "https": proxy_url
            }

        try:
            response = requests.post(full_url, headers=headers, json=payload, proxies=proxies, timeout=60)
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return (content,)
            else:
                return (f"Error: API response format unexpected. Response: {json.dumps(result)}",)

        except Exception as e:
            return (f"Error calling OpenAI API: {str(e)}",)

# 节点映射字典
NODE_CLASS_MAPPINGS = {
    "OpenAIAPINode": OpenAINode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIAPINode": "OpenAI API 连接 / OpenAI API Connector"
}
