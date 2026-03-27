import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import folder_paths
import os
import json

class RuiMaskPreview:
    """
    遮罩预览节点：将遮罩以半透明彩色形式叠加到图像上进行可视化预览
    """
    
    COLOR_MAP = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
        "white": (1.0, 1.0, 1.0),
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mask_color": (
                    ["红色 / Red", "绿色 / Green", "蓝色 / Blue", "黄色 / Yellow", 
                     "青色 / Cyan", "品红 / Magenta", "白色 / White"],
                    {"default": "红色 / Red"}
                ),
            },
            "optional": {
                "opacity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像 / Image",)
    FUNCTION = "preview_mask"
    CATEGORY = "Rui-Node🐶/遮罩处理🎭"
    OUTPUT_NODE = True
    
    def preview_mask(self, image, mask, mask_color, opacity=0.5):
        """
        将遮罩以半透明彩色形式叠加到图像上
        
        参数:
            image: 输入图像张量 (N, H, W, C)
            mask: 输入遮罩张量 (N, H, W) 或 (H, W)
            mask_color: 遮罩显示颜色（中英双语字符串）
            opacity: 不透明度 (0.0-1.0)
        
        返回:
            合成后的图像张量和预览信息
        """
        color_mapping = {
            "红色 / Red": "red",
            "绿色 / Green": "green",
            "蓝色 / Blue": "blue",
            "黄色 / Yellow": "yellow",
            "青色 / Cyan": "cyan",
            "品红 / Magenta": "magenta",
            "白色 / White": "white",
        }
        
        color_key = color_mapping.get(mask_color, "red")
        color_rgb = self.COLOR_MAP[color_key]
        
        batch, height, width, channels = image.shape
        
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        
        mask_batch, mask_height, mask_width = mask.shape
        
        if mask_height != height or mask_width != width:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(
                mask, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
            mask = mask.squeeze(1)
        
        if mask_batch == 1 and batch > 1:
            mask = mask.repeat(batch, 1, 1)
        elif mask_batch != batch:
            min_batch = min(mask_batch, batch)
            mask = mask[:min_batch]
            image = image[:min_batch]
            batch = min_batch
            print(f"警告: 遮罩批次数({mask_batch})与图像批次数({batch})不匹配，已截取为{min_batch}")
        
        mask = torch.clamp(mask, 0.0, 1.0)
        
        if channels > 3:
            image = image[:, :, :, :3]
        
        mask_expanded = mask.unsqueeze(-1)
        
        color_tensor = torch.tensor(
            color_rgb, 
            dtype=image.dtype, 
            device=image.device
        ).view(1, 1, 1, 3)
        
        color_layer = mask_expanded * color_tensor
        
        alpha = mask_expanded * opacity
        
        output = image * (1 - alpha) + color_layer * opacity
        
        output = torch.clamp(output, 0.0, 1.0)
        
        results = self.save_images(output)
        
        return {
            "ui": {"images": results},
            "result": (output,)
        }
    
    def save_images(self, images):
        """
        保存图像供预览使用
        
        参数:
            images: 图像张量 (N, H, W, C)
        
        返回:
            包含图像信息的列表
        """
        results = []
        
        output_dir = folder_paths.get_temp_directory()
        
        for i, image_tensor in enumerate(images):
            img_np = image_tensor.cpu().numpy()
            
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            
            img_pil = Image.fromarray(img_np, 'RGB')
            
            filename = f"mask_preview_{i:05d}.png"
            filepath = os.path.join(output_dir, filename)
            
            img_pil.save(filepath, compress_level=4)
            
            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "temp"
            })
        
        return results

NODE_CLASS_MAPPINGS = {
    "RuiMaskPreview": RuiMaskPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiMaskPreview": "遮罩预览 / Mask Preview"
}
