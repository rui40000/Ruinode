import torch
import numpy as np
from PIL import Image

class FlipNode:
    """
    图像翻转节点：可以将图像进行左右或上下翻转
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "flip_direction": (["水平", "垂直"], {
                    "default": "水平"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "flip_image"
    CATEGORY = "Rui-Node🐶/图像调节🎨"

    def flip_image(self, image, flip_direction):
        """
        翻转输入图像
        
        参数:
            image: 输入图像张量 (B, H, W, C) 格式
            flip_direction: 翻转方向，"水平"或"垂直"
        
        返回:
            翻转后的图像张量
        """
        # 将图像从 PyTorch 张量转换为 PIL 图像进行处理
        batch_size = image.shape[0]
        result = []
        
        for i in range(batch_size):
            # 将单个图像从 PyTorch 张量转换为 NumPy 数组
            # ComfyUI 中图像格式为 BHWC，值范围为 0-1
            img_np = image[i].cpu().numpy()
            
            # 确保值范围在 0-1 之间
            img_np = np.clip(img_np, 0, 1)
            
            # 转换为 PIL 图像 (值范围 0-255)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8), 'RGB')
            
            # 根据选择的方向翻转图像
            if flip_direction == "水平":
                flipped_img = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            else:  # 垂直翻转
                flipped_img = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
            
            # 将处理后的图像转回 NumPy 数组，然后转为 PyTorch 张量
            flipped_np = np.array(flipped_img).astype(np.float32) / 255.0
            flipped_tensor = torch.from_numpy(flipped_np)
            
            result.append(flipped_tensor)
        
        # 将结果堆叠为批次
        return (torch.stack(result),)

# 节点映射字典，用于 ComfyUI 注册节点
NODE_CLASS_MAPPINGS = {
    "ImageFlip": FlipNode
}

# 节点显示名称映射，用于在 UI 中显示友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFlip": "图像翻转 / Image Flip"
}