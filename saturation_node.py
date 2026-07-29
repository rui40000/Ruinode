import torch
import numpy as np
from PIL import Image, ImageEnhance

class SaturationNode:
    """
    调整图像饱和度的节点
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "要调整饱和度的图像。"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "饱和度系数：\n"
                               "0.0 = 完全去色（黑白）\n"
                               "1.0 = 保持原样\n"
                               "1.0~2.0 = 色彩逐渐浓郁，超过 2.0 容易溢出失真"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_saturation"
    CATEGORY = "Rui-Node🐶/图像调节🎨"

    def adjust_saturation(self, image, saturation):
        """
        调整输入图像的饱和度
        
        参数:
            image: 输入图像张量 (B, H, W, C) 格式
            saturation: 饱和度调整系数，1.0为原始饱和度
        
        返回:
            调整后的图像张量
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
            
            # 使用 PIL 的 ImageEnhance.Color 调整饱和度
            enhancer = ImageEnhance.Color(img_pil)
            enhanced_img = enhancer.enhance(saturation)
            
            # 将处理后的图像转回 NumPy 数组，然后转为 PyTorch 张量
            enhanced_np = np.array(enhanced_img).astype(np.float32) / 255.0
            enhanced_tensor = torch.from_numpy(enhanced_np)
            
            result.append(enhanced_tensor)
        
        # 将结果堆叠为批次
        return (torch.stack(result),)

# 节点映射字典，用于 ComfyUI 注册节点
NODE_CLASS_MAPPINGS = {
    "SaturationAdjustment": SaturationNode
}

# 节点显示名称映射，用于在 UI 中显示友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaturationAdjustment": "调整饱和度 / Saturation Adjustment"
}