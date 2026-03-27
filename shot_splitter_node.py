import re

class ShotSplitterNode:
    """
    镜头分词器节点：将输入的多分镜描述脚本拆分成独立的分镜描述
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
            "optional": {
                "start_shot_num": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
                "shot_count": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                }),
            },
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("shot_descriptions", "summary")
    FUNCTION = "split_shots"
    CATEGORY = "Rui-Node🐶/文本处理📝"

    def split_shots(self, input_text, start_shot_num=0, shot_count=0):
        """
        将输入的多分镜描述脚本拆分成独立的分镜描述
        
        参数:
            input_text: 输入的多分镜描述脚本
        
        返回:
            shot_descriptions: 拆分后的分镜描述列表（Python列表格式的字符串）
            summary: 总结信息
        """
        # 使用正则表达式匹配所有分镜内容
        # 匹配<SHOT_XXX>和</SHOT_XXX>之间的内容，其中XXX是数字
        pattern = r'<SHOT_(\d+)>([\s\S]*?)</SHOT_\1>'
        matches = re.findall(pattern, input_text)
        
        # 提取分镜描述
        shot_descriptions = []
        for shot_num, content in matches:
            # 去除开头和结尾的空白字符，但保留内部段落结构
            content = content.strip()
            shot_descriptions.append(content)
        
        # 根据开始和结束分镜编号筛选分镜
        filtered_shots = []
        all_shot_nums = [int(shot_num) for shot_num, _ in matches]
        
        # 如果用户没有输入开始编号和导出数量（都为0），则导出所有分镜
        if start_shot_num == 0 and shot_count == 0:
            filtered_shots = shot_descriptions
        else:
            # 如果没有设置开始编号，则从第一个分镜开始
            if start_shot_num == 0:
                start_shot_num = min(all_shot_nums) if all_shot_nums else 0
            
            # 计算结束编号
            end_shot_num = start_shot_num + shot_count - 1 if shot_count > 0 else max(all_shot_nums)
            
            # 筛选指定范围内的分镜
            for i, (shot_num, content) in enumerate(matches):
                if start_shot_num <= int(shot_num) <= end_shot_num:
                    filtered_shots.append(shot_descriptions[i])
        
        # 生成总结信息
        if start_shot_num > 0 or shot_count > 0:
            end_num = start_shot_num + shot_count - 1 if shot_count > 0 else max(all_shot_nums)
            summary = f"一共拆分成{len(shot_descriptions)}个分镜，导出了{len(filtered_shots)}个分镜（从{start_shot_num}开始，共{shot_count if shot_count > 0 else '全部'}个）。"
        else:
            summary = f"一共拆分成{len(shot_descriptions)}个分镜。"
        
        # 直接返回Python列表，不转换为字符串
        return (filtered_shots, summary)

# 节点映射字典，用于 ComfyUI 注册节点
NODE_CLASS_MAPPINGS = {
    "ShotSplitter": ShotSplitterNode
}

# 节点显示名称映射，用于在 UI 中显示友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShotSplitter": "镜头分词器 / Shot Splitter"
}