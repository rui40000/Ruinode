import torch
import numpy as np
from PIL import Image
import os

class LoadImageByPathNode:
    """
    按路径加载图像的节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {
                    "default": "E:\\ComfyUIModels\\input\\10\\1.png",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_image_by_path"
    CATEGORY = "Rui-Node🐶/文件存储与加载📁"

    def load_image_by_path(self, image_path):
        """
        根据指定路径加载图像
        
        参数:
            image_path: 图像文件的完整路径
        
        返回:
            加载的图像张量
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            # 使用 PIL 加载图像
            img_pil = Image.open(image_path)
            
            # 确保图像为 RGB 模式
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            
            # 将 PIL 图像转换为 NumPy 数组
            img_np = np.array(img_pil).astype(np.float32) / 255.0
            
            # 转换为 PyTorch 张量并添加批次维度
            # ComfyUI 期望的格式为 BHWC
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            return (img_tensor,)
            
        except Exception as e:
            # 如果加载失败，创建一个默认的黑色图像
            print(f"加载图像失败: {str(e)}")
            # 创建一个 512x512 的黑色图像作为默认值
            default_img = np.zeros((512, 512, 3), dtype=np.float32)
            default_tensor = torch.from_numpy(default_img).unsqueeze(0)
            return (default_tensor,)

# 节点映射字典，用于 ComfyUI 注册节点
NODE_CLASS_MAPPINGS = {
    "LoadImageByPath": LoadImageByPathNode
}

# 节点显示名称映射，用于在 UI 中显示友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageByPath": "按路径加载图像 / Load Image By Path"
}