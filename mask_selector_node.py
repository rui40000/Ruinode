import torch
import json

class MaskSelector:
    """
    遮罩筛选节点：
    - 对输入遮罩进行排序（按面积、从左到右、从上到下）
    - 输出指定编号的遮罩、剩余遮罩的合并结果，以及 JSON 信息
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "sort_method": (["按面积排序 / By Area", "从左到右 / Left to Right", "从上到下 / Top to Bottom"], {
                    "default": "按面积排序 / By Area"
                }),
                "index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "STRING")
    RETURN_NAMES = ("选中遮罩 / Selected", "剩余遮罩 / Remaining", "信息 / Info")
    FUNCTION = "select_mask"
    CATEGORY = "Rui-Node🐶/遮罩处理🎭"

    def select_mask(self, masks, sort_method, index):
        """
        主处理流程：
        - 标准化输入形状为 (N, H, W)
        - 计算每个遮罩的面积与质心
        - 按指定规则排序
        - 选择指定编号并返回三个输出
        """
        # 标准化输入形状
        if masks.dim() == 2:
            # (H, W) -> (1, H, W)
            masks = masks.unsqueeze(0)
        elif masks.dim() == 3:
            pass  # (N, H, W)
        else:
            # 非法形状，返回空结果
            H = W = 0
            empty = torch.zeros((1, H, W), dtype=torch.float32)
            info = json.dumps({"error": "invalid mask shape"}, ensure_ascii=False)
            return (empty, empty, info)

        N, H, W = masks.shape

        # 计算面积与质心
        # 使用阈值 0.5 判定非零像素
        bin_masks = (masks > 0.5)

        areas = []
        centers = []
        all_empty = True
        for i in range(N):
            nz = torch.nonzero(bin_masks[i], as_tuple=False)
            area = nz.shape[0]
            areas.append(int(area))
            if area > 0:
                all_empty = False
                # 质心坐标：x 为列索引平均，y 为行索引平均
                y_mean = float(nz[:, 0].float().mean().item())
                x_mean = float(nz[:, 1].float().mean().item())
                centers.append((x_mean, y_mean))
            else:
                # 空遮罩：用占位中心使其排序在末尾
                centers.append((float('inf'), float('inf')))

        # 映射中英双语排序选项到内部代码
        method_map = {
            "按面积排序 / By Area": "by_area",
            "从左到右 / Left to Right": "left_to_right",
            "从上到下 / Top to Bottom": "top_to_bottom",
            # 兼容老值
            "by_area": "by_area",
            "left_to_right": "left_to_right",
            "top_to_bottom": "top_to_bottom",
        }
        internal_method = method_map.get(sort_method, "by_area")

        # 排序索引
        if internal_method == "by_area":
            # 面积从大到小
            sort_key = [(areas[i], -i) for i in range(N)]
            order = sorted(range(N), key=lambda i: sort_key[i], reverse=True)
        elif internal_method == "left_to_right":
            # 按质心 X 升序
            sort_key = [centers[i][0] for i in range(N)]
            order = sorted(range(N), key=lambda i: sort_key[i])
        elif internal_method == "top_to_bottom":
            # 按质心 Y 升序
            sort_key = [centers[i][1] for i in range(N)]
            order = sorted(range(N), key=lambda i: sort_key[i])
        else:
            # 未知排序方式，默认按面积
            sort_key = [(areas[i], -i) for i in range(N)]
            order = sorted(range(N), key=lambda i: sort_key[i], reverse=True)

        # 1-based 索引选择与夹取
        index_clamped = False
        target_idx = index - 1
        if target_idx < 0:
            target_idx = 0
            index_clamped = True
        if target_idx >= N:
            target_idx = N - 1
            index_clamped = True

        selected_i = order[target_idx]
        selected = masks[selected_i].unsqueeze(0)  # (1, H, W)

        # 合并剩余遮罩：逻辑或
        remaining_indices = [i for i in order if i != selected_i]
        if len(remaining_indices) == 0:
            remaining = torch.zeros((1, H, W), dtype=masks.dtype, device=masks.device)
        else:
            remaining_stack = masks[remaining_indices]  # (M, H, W)
            # 使用 max 合并为逻辑或
            remaining = torch.max(remaining_stack, dim=0).values.unsqueeze(0)

        # JSON 信息（包含中文注释的字符串）
        selected_area = areas[selected_i]
        sel_center = centers[selected_i]
        # 对空遮罩设定中心为 [0, 0]
        if sel_center[0] == float('inf') or sel_center[1] == float('inf'):
            sel_center_out = [0, 0]
        else:
            sel_center_out = [int(round(sel_center[0])), int(round(sel_center[1]))]

        info_obj = {
            "total_masks": N,                    # 遮罩总数：检测到的遮罩总数量
            "selected_index": index,             # 选中编号：用户指定输出的遮罩编号（原始输入）
            "sort_method": internal_method,      # 排序方式：当前使用的排序规则代码
            "selected_area": selected_area,      # 选中面积：选中遮罩的像素面积
            "selected_center": sel_center_out,   # 选中中心点：选中遮罩的质心坐标 [x, y]
            "index_clamped": index_clamped       # 编号越界：若编号超出范围被自动修正则为 true
        }
        if all_empty:
            info_obj["warning"] = "所有遮罩为空 / All masks empty"

        # 生成带注释的 JSON 字符串
        # 注意：标准 JSON 不支持注释，这里按照需求输出 JSON 风格字符串并附带注释
        # 若后续需严格 JSON，可移除注释并使用 json.dumps(info_obj, ensure_ascii=False)
        info_lines = [
            "{",
            f'  "total_masks": {info_obj["total_masks"]},           // 遮罩总数：检测到的遮罩总数量',
            f'  "selected_index": {info_obj["selected_index"]},        // 选中编号：用户指定输出的遮罩编号',
            f'  "sort_method": "{info_obj["sort_method"]}",   // 排序方式：当前使用的排序规则代码',
            f'  "selected_area": {info_obj["selected_area"]},     // 选中面积：选中遮罩的像素面积',
            f'  "selected_center": {json.dumps(info_obj["selected_center"], ensure_ascii=False)},  // 选中中心点：选中遮罩的质心坐标 [x, y]',
            f'  "index_clamped": {"true" if info_obj["index_clamped"] else "false"}      // 编号越界：如果用户输入的编号超出范围被自动修正则为 true'
        ]
        if "warning" in info_obj:
            info_lines.append(f'  ,"warning": "{info_obj["warning"]}"  // 警告：所有遮罩为空')
        info_lines.append("}")
        info_str = "\n".join(info_lines)

        return (selected, remaining, info_str)


# 节点映射字典，用于 ComfyUI 注册节点
NODE_CLASS_MAPPINGS = {
    "MaskSelector": MaskSelector
}

# 节点显示名称映射，用于在 UI 中显示友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskSelector": "遮罩筛选 / Mask Selector"
}