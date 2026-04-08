import torch
import numpy as np
import scipy.ndimage

class SpriteSplitter:
    """
    从白色/浅色背景的合图中自动拆分独立素材元素。
    去除背景后，通过连通区域检测将每个独立元素裁剪为单独的图像输出。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "最小面积过滤（像素数）": ("INT", {
                    "default": 100,
                    "min": 1,
                    "max": 50000,
                    "step": 1,
                    "tooltip": "最小面积（像素数）：面积小于此值的连通区域将被过滤，避免噪点碎片。"
                }),
                "裁剪边距": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 50,
                    "step": 1,
                    "tooltip": "裁剪边距（像素）：每个元素裁剪时在包围盒外额外保留的像素边距。"
                }),
                "排序方式": (["从左到右-从上到下", "从上到下-从左到右", "面积从大到小", "面积从小到大"],{
                    "default": "从左到右-从上到下",
                    "tooltip": "输出排序方式：控制拆分后元素的排列顺序。"
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "随机种子：仅用于强制重新执行节点，不影响实际拆分结果。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像列表",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split_sprites"
    CATEGORY = "Rui-Node🐶/图像调节🎨"
    DESCRIPTION = "Sprite Splitter - 从白色背景合图中自动拆分独立素材元素"

    def split_sprites(self, **kwargs):
        图像 = kwargs.get("图像")
        最小面积过滤_像素数 = kwargs.get("最小面积过滤（像素数）")
        裁剪边距 = kwargs.get("裁剪边距")
        排序方式 = kwargs.get("排序方式")
        """
        主处理函数。
        """
        # 取 batch 中的第一张图
        img_tensor = 图像[0]  # [H, W, C]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        h, w, c = img_np.shape

        # 确保图像有 Alpha 通道
        if c == 3:
            # 如果输入只有 RGB，说明用户可能没有提供透明背景的图
            # 但既然去除了内置的去背逻辑，我们只能假设纯黑 (0,0,0) 或用户需要自行提供 RGBA
            # 为了兼容性，这里我们默认将全图设为不透明，或者如果有单独 Mask 处理的话...
            # 建议用户输入带有透明通道的图像。
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_np
            rgba[:, :, 3] = 255  # 全不透明，将导致无法拆分，所以最好输入带Alpha的图
        else:
            rgba = img_np.copy()

        # ========== 连通区域检测 (使用 scipy.ndimage 加速) ==========
        alpha = (rgba[:, :, 3] > 0).astype(np.uint8)
        
        # 定义 4-连通 的结构元素
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
        
        # 使用 scipy.ndimage.label 找出连通区域
        labels, num_features = scipy.ndimage.label(alpha, structure=structure)
        
        bboxes = []
        if num_features > 0:
            # 使用 scipy.ndimage.find_objects 快速获取包围盒
            slices = scipy.ndimage.find_objects(labels)
            
            # 使用 numpy.bincount 快速计算每个标签的面积
            areas = np.bincount(labels.ravel())
            
            for i, slc in enumerate(slices):
                if slc is not None:
                    label_idx = i + 1
                    area = areas[label_idx]
                    
                    min_y, max_y = slc[0].start, slc[0].stop - 1
                    min_x, max_x = slc[1].start, slc[1].stop - 1
                    
                    bboxes.append({
                        'label': label_idx,
                        'min_x': min_x, 'min_y': min_y,
                        'max_x': max_x, 'max_y': max_y,
                        'area': area
                    })

        # ========== Step 3: 过滤和排序 ==========
        valid_bboxes = [b for b in bboxes if b['area'] >= 最小面积过滤_像素数]

        if 排序方式 == "从左到右-从上到下":
            row_height = max(60, h // 15)  # 自适应行高
            valid_bboxes.sort(key=lambda b: (b['min_y'] // row_height, b['min_x']))
        elif 排序方式 == "从上到下-从左到右":
            col_width = max(60, w // 15)
            valid_bboxes.sort(key=lambda b: (b['min_x'] // col_width, b['min_y']))
        elif 排序方式 == "面积从大到小":
            valid_bboxes.sort(key=lambda b: -b['area'])
        elif 排序方式 == "面积从小到大":
            valid_bboxes.sort(key=lambda b: b['area'])

        # ========== Step 4: 裁剪并输出 ==========
        result_images = []

        for b in valid_bboxes:
            sx = max(0, b['min_x'] - 裁剪边距)
            sy = max(0, b['min_y'] - 裁剪边距)
            ex = min(w, b['max_x'] + 裁剪边距 + 1)
            ey = min(h, b['max_y'] + 裁剪边距 + 1)
            
            # 使用 numpy 切片快速裁剪
            sprite_rgba = rgba[sy:ey, sx:ex].copy()
            sprite_labels = labels[sy:ey, sx:ex]
            
            # 只保留当前标签的像素，其他设为透明
            mask = (sprite_labels != b['label'])
            sprite_rgba[mask] = 0

            # 转为 ComfyUI IMAGE tensor: [1, H, W, C], float32, range [0,1]
            # ComfyUI 标准 IMAGE 是 RGB (3通道)，我们输出 RGBA 以保留透明度
            # 但标准IMAGE是3通道，所以将透明区域设为白色并输出RGB
            alpha_f = sprite_rgba[:, :, 3:4].astype(np.float32) / 255.0
            sprite_rgb = (sprite_rgba[:, :, :3].astype(np.float32) / 255.0) * alpha_f + (1.0 - alpha_f)

            tensor = torch.from_numpy(sprite_rgb).unsqueeze(0)  # [1, H, W, 3]
            result_images.append(tensor)

        # 如果没有检测到元素，返回原图
        if not result_images:
            result_images.append(图像[:1])

        return (result_images,)


class SpriteSplitterRGBA:
    """
    与 SpriteSplitter 功能相同，但输出带 Alpha 通道的 RGBA 图像。
    透明区域保持透明（而非白色填充）。
    需要下游节点支持 4 通道图像（如 SaveImageWithAlpha 等）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "最小面积过滤（像素数）": ("INT", {
                    "default": 100, "min": 1, "max": 50000, "step": 1,
                    "tooltip": "最小面积过滤。"
                }),
                "裁剪边距": ("INT", {
                    "default": 2, "min": 0, "max": 50, "step": 1,
                    "tooltip": "裁剪边距（像素）。"
                }),
                "排序方式": (["从左到右-从上到下", "从上到下-从左到右", "面积从大到小", "面积从小到大"],{
                    "default": "从左到右-从上到下",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "随机种子：仅用于强制重新执行节点，不影响实际拆分结果。"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像列表", "遮罩列表")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "split_sprites_rgba"
    CATEGORY = "Rui-Node🐶/图像调节🎨"
    DESCRIPTION = "Sprite Splitter RGBA - 拆分素材并输出带透明通道的图像 + Mask"

    def split_sprites_rgba(self, **kwargs):
        图像 = kwargs.get("图像")
        最小面积过滤_像素数 = kwargs.get("最小面积过滤（像素数）")
        裁剪边距 = kwargs.get("裁剪边距")
        排序方式 = kwargs.get("排序方式")
        img_tensor = 图像[0]
        img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        h, w, c = img_np.shape

        if c == 3:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = img_np
            rgba[:, :, 3] = 255
        else:
            rgba = img_np.copy()

        alpha = (rgba[:, :, 3] > 0).astype(np.uint8)
        
        # 定义 4-连通 的结构元素
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
        
        # 使用 scipy.ndimage.label 找出连通区域
        labels, num_features = scipy.ndimage.label(alpha, structure=structure)
        
        bboxes = []
        if num_features > 0:
            # 使用 scipy.ndimage.find_objects 快速获取包围盒
            slices = scipy.ndimage.find_objects(labels)
            
            # 使用 numpy.bincount 快速计算每个标签的面积
            areas = np.bincount(labels.ravel())
            
            for i, slc in enumerate(slices):
                if slc is not None:
                    label_idx = i + 1
                    area = areas[label_idx]
                    
                    min_y, max_y = slc[0].start, slc[0].stop - 1
                    min_x, max_x = slc[1].start, slc[1].stop - 1
                    
                    bboxes.append({
                        'label': label_idx,
                        'min_x': min_x, 'min_y': min_y,
                        'max_x': max_x, 'max_y': max_y,
                        'area': area
                    })

        valid_bboxes = [b for b in bboxes if b['area'] >= 最小面积过滤_像素数]

        if 排序方式 == "从左到右-从上到下":
            row_height = max(60, h // 15)
            valid_bboxes.sort(key=lambda b: (b['min_y'] // row_height, b['min_x']))
        elif 排序方式 == "从上到下-从左到右":
            col_width = max(60, w // 15)
            valid_bboxes.sort(key=lambda b: (b['min_x'] // col_width, b['min_y']))
        elif 排序方式 == "面积从大到小":
            valid_bboxes.sort(key=lambda b: -b['area'])
        elif 排序方式 == "面积从小到大":
            valid_bboxes.sort(key=lambda b: b['area'])

        result_images = []
        result_masks = []

        for b in valid_bboxes:
            sx = max(0, b['min_x'] - 裁剪边距)
            sy = max(0, b['min_y'] - 裁剪边距)
            ex = min(w, b['max_x'] + 裁剪边距 + 1)
            ey = min(h, b['max_y'] + 裁剪边距 + 1)
            
            # 使用 numpy 切片快速裁剪
            sprite_rgba = rgba[sy:ey, sx:ex].copy()
            sprite_labels = labels[sy:ey, sx:ex]
            
            # 只保留当前标签的像素，其他设为透明
            mask = (sprite_labels != b['label'])
            sprite_rgba[mask] = 0

            # IMAGE: RGB [1, H, W, 3]
            sprite_rgb = sprite_rgba[:, :, :3].astype(np.float32) / 255.0
            img_t = torch.from_numpy(sprite_rgb).unsqueeze(0)
            result_images.append(img_t)

            # MASK: [1, H, W], 1.0 = 不透明, 0.0 = 透明
            mask_np = sprite_rgba[:, :, 3].astype(np.float32) / 255.0
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)
            result_masks.append(mask_t)

        if not result_images:
            result_images.append(图像[:1])
            h0, w0 = 图像.shape[1], 图像.shape[2]
            result_masks.append(torch.ones(1, h0, w0))

        return (result_images, result_masks)


# ======== 节点注册 ========
NODE_CLASS_MAPPINGS = {
    "RuiSpriteSplitter": SpriteSplitter,
    "RuiSpriteSplitterRGBA": SpriteSplitterRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiSpriteSplitter": "素材拆分 / Sprite Splitter",
    "RuiSpriteSplitterRGBA": "素材拆分(带透明通道) / Sprite Splitter RGBA",
}
