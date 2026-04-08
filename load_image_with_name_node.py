import os
import torch
import numpy as np
import hashlib
from PIL import Image, ImageOps, ImageSequence
import folder_paths

class LoadImageWithNameNode:
    """
    基础功能与 ComfyUI 的通用图像加载节点相同，但额外输出图像的文件名（不带后缀）。
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "Rui-Node🐶/文件存储与加载📁"

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "filename")
    FUNCTION = "load_image"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        img = Image.open(image_path)
        
        # 提取不带后缀的文件名
        filename = os.path.splitext(os.path.basename(image_path))[0]
        
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            
            image_rgba = i.convert("RGBA")
            image_rgb = image_rgba.convert("RGB")
            
            image_np = np.array(image_rgb).astype(np.float32) / 255.0
            output_images.append(image_np)
            
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - mask
            else:
                mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
            output_masks.append(mask)

        if len(output_images) > 1:
            output_image = torch.from_numpy(np.stack(output_images))
            output_mask = torch.from_numpy(np.stack(output_masks))
        else:
            output_image = torch.from_numpy(output_images[0])[None,]
            output_mask = torch.from_numpy(output_masks[0])[None,]

        return (output_image, output_mask, filename)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

NODE_CLASS_MAPPINGS = {
    "LoadImageWithName": LoadImageWithNameNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithName": "加载图像(带文件名) / Load Image With Name"
}
