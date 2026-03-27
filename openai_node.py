import torch
import numpy as np
import requests
import json
import base64
import io
import os
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
                "api_url": ("STRING", {
                    "default": "https://api.openai.com/v1/chat/completions",
                    "multiline": False
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
                # max_tokens, temperature 等常用参数可以根据需要添加，这里保持精简
            },
            "optional": {
                "image": ("IMAGE",),
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

    def generate_content(self, api_url, api_key, model, system_prompt, user_prompt, seed, image=None, proxy_url=""):
        """
        调用 OpenAI API 生成内容
        """
        
        # 准备消息列表
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        user_content = []
        
        # 添加用户文本提示词
        if user_prompt:
            user_content.append({
                "type": "text",
                "text": user_prompt
            })
            
        # 处理图像输入
        if image is not None:
            # 获取批次中的第一张图像
            img_tensor = image[0]
            
            # 将 Tensor 转换为 PIL Image
            img_np = img_tensor.cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
            
            # 将图像转换为 base64
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # 添加图像内容
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_base64}"
                }
            })
        
        # 如果 user_content 为空，且没有图像，至少添加一个空文本以防 API 报错
        if not user_content:
             user_content.append({
                "type": "text",
                "text": " " 
            })

        # 构造用户消息
        # 注意：对于不支持多模态的模型（如 gpt-3.5-turbo），发送 image_url 可能会报错
        # 但遵循“符合最新规范”的要求，我们默认使用 content list 结构
        # 如果模型不支持 list content，可以尝试回退到纯字符串（但这会丢失图片）
        # 这里为了保持代码简洁，我们始终使用 list 结构，依赖用户选择支持 vision 的模型或仅输入文本
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # 构造请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 构造请求体
        payload = {
            "model": model,
            "messages": messages,
            "seed": seed,
            # 可以添加 temperature 等参数
        }
        
        # 处理代理设置
        proxies = None
        if proxy_url and proxy_url.strip():
            proxies = {
                "http": proxy_url,
                "https": proxy_url
            }

        try:
            # 发送请求
            response = requests.post(api_url, headers=headers, json=payload, proxies=proxies, timeout=60)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 提取生成的文本
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
