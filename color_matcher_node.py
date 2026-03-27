import numpy as np
import torch

def to_numpy_uint8(img_t):  # [H,W,C] float(0..1) -> np.uint8
    img = img_t.detach().cpu().numpy()
    img = (np.clip(img, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return img

def to_tensor_image(img_np):  # np.uint8 HWC -> torch [H,W,C] float(0..1)
    t = torch.from_numpy(img_np.astype(np.float32) / 255.0)
    return t

def histogram_match(source, template):
    # 逐通道直方图匹配
    src = source.copy()
    for c in range(src.shape[2]):
        s = src[..., c].ravel()
        t = template[..., c].ravel()
        s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(t, return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64) / s.size
        t_quantiles = np.cumsum(t_counts).astype(np.float64) / t.size
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        src[..., c] = interp_t_values[bin_idx].reshape(src.shape[:2])
    return src

class ColorMatcherNode:
    """颜色匹配节点：将移动图像的颜色分布匹配到参考图像"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "moving_image": ("IMAGE",),
                "match_method": (["histogram", "mean_std", "none"], {"default": "histogram"}),
                "blend_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("颜色匹配后图像", "匹配信息")
    FUNCTION = "match_colors"
    CATEGORY = "Rui-Node🐶/图像调节🎨"

    def match_colors(self, reference_image, moving_image, match_method, blend_factor):
        B = reference_image.shape[0]
        assert moving_image.shape[0] == B, "batch 大小必须一致"

        out_images = []
        match_info = []

        for i in range(B):
            ref_img = reference_image[i]  # [H,W,C]
            mov_img = moving_image[i]

            if match_method == "none":
                match_info.append(f"Batch {i}: 跳过颜色匹配")
                out_images.append(mov_img.unsqueeze(0))
                continue

            ref_np = to_numpy_uint8(ref_img)
            mov_np = to_numpy_uint8(mov_img)

            if match_method == "histogram":
                # 直方图匹配
                matched_np = histogram_match(mov_np, ref_np)
                method_name = "直方图匹配"
            elif match_method == "mean_std":
                # 均值标准差匹配
                matched_np = self.mean_std_match(mov_np, ref_np)
                method_name = "均值标准差匹配"
            else:
                matched_np = mov_np
                method_name = "无匹配"

            # 混合原图和匹配后的图像
            if blend_factor < 1.0:
                matched_np = (matched_np * blend_factor + mov_np * (1 - blend_factor)).astype(np.uint8)

            match_info.append(f"Batch {i}: {method_name}，混合系数: {blend_factor:.2f}")
            matched_tensor = to_tensor_image(matched_np)
            out_images.append(matched_tensor.unsqueeze(0))

        matched_images = torch.cat(out_images, dim=0)
        info_str = " | ".join(match_info)

        return (matched_images, info_str)

    def mean_std_match(self, source, template):
        """均值标准差颜色匹配"""
        src = source.astype(np.float32)
        tpl = template.astype(np.float32)
        
        matched = src.copy()
        for c in range(src.shape[2]):
            src_mean = np.mean(src[..., c])
            src_std = np.std(src[..., c])
            tpl_mean = np.mean(tpl[..., c])
            tpl_std = np.std(tpl[..., c])
            
            if src_std > 0:
                matched[..., c] = (src[..., c] - src_mean) * (tpl_std / src_std) + tpl_mean
        
        return np.clip(matched, 0, 255).astype(np.uint8)

NODE_CLASS_MAPPINGS = {
    "ColorMatcher": ColorMatcherNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorMatcher": "颜色匹配器 / Color Matcher"
}