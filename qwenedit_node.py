import torch
import numpy as np
import requests
import json
import base64
import io
from PIL import Image
import random

class QwenEditNode:
    """
    使用阿里云千问编辑模型API进行图像生成的节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "base_url": ("STRING", {
                    "default": "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image-generation/generation",
                    "multiline": False
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647
                }),
                "control_mode": (["reference", "sketch", "scribble", "pose", "canny", "depth", "hed", "mlsd", "normal", "seg"], {
                    "default": "reference"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 8
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 2048,
                    "step": 8
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "Rui-Node🐶/AI模型🤖"

    def generate_image(self, image1, image2, image3, image4, api_key, base_url, seed, control_mode, width, height):
        """
        使用千问编辑模型API生成图像
        
        参数:
            image1-4: 输入图像张量 (B, H, W, C) 格式
            api_key: 阿里云API密钥
            base_url: API基础URL
            seed: 随机种子值，-1表示随机生成
            control_mode: 控制模式
            width: 输出图像宽度
            height: 输出图像高度
        
        返回:
            生成的图像张量
        """
        # 如果seed为-1，则随机生成种子
        if seed == -1:
            seed = random.randint(0, 2147483647)
            
        # 准备图像数据
        images = [image1, image2, image3, image4]
        image_data = []
        
        for i, img in enumerate(images):
            if img is None or img.shape[0] == 0:
                continue
                
            # 取批次中的第一张图像
            img_np = img[0].cpu().numpy()
            
            # 确保值范围在 0-1 之间
            img_np = np.clip(img_np, 0, 1)
            
            # 转换为 PIL 图像 (值范围 0-255)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
            
            # 转换为base64编码
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 添加到图像数据列表
            image_data.append({
                "image": img_base64,
                "control_type": control_mode
            })
        
        # 准备API请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "qwen-vl-plus",
            "input": {
                "images": image_data
            },
            "parameters": {
                "seed": seed,
                "width": width,
                "height": height
            }
        }
        
        try:
            # 发送API请求
            response = requests.post(base_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            # 检查是否成功
            if "output" in result and "images" in result["output"] and len(result["output"]["images"]) > 0:
                # 获取生成的图像
                generated_image_base64 = result["output"]["images"][0]
                
                # 解码base64图像
                image_bytes = base64.b64decode(generated_image_base64)
                img_pil = Image.open(io.BytesIO(image_bytes))
                
                # 确保图像为 RGB 模式
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                
                # 转换为 NumPy 数组
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                
                # 转换为 PyTorch 张量并添加批次维度
                img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                
                return (img_tensor,)
            else:
                # 如果响应中没有图像，返回错误信息
                print(f"API响应中没有图像: {result}")
                # 创建一个默认的黑色图像作为返回值
                default_img = np.zeros((height, width, 3), dtype=np.float32)
                default_tensor = torch.from_numpy(default_img).unsqueeze(0)
                return (default_tensor,)
                
        except Exception as e:
            # 如果API请求失败，打印错误信息并返回默认图像
            print(f"API请求失败: {str(e)}")
            # 创建一个默认的黑色图像作为返回值
            default_img = np.zeros((height, width, 3), dtype=np.float32)
            default_tensor = torch.from_numpy(default_img).unsqueeze(0)
            return (default_tensor,)

# 节点映射字典，用于 ComfyUI 注册节点
NODE_CLASS_MAPPINGS = {
    "QwenEditImageGeneration": QwenEditNode
}

# 节点显示名称映射，用于在 UI 中显示友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenEditImageGeneration": "千问编辑图像生成 / Qwen Edit Image Generation"
}