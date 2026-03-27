import re
import torch

class DialogueExtractorNode:
    """
    对白提取器节点：从分镜描述中提取旁白/对白部分
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
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("dialogues", "summary")
    FUNCTION = "extract_dialogues"
    CATEGORY = "Rui-Node🐶/文本处理📝"

    def extract_dialogues(self, input_text):
        """
        从分镜描述中提取旁白/对白部分
        
        参数:
            input_text: 输入的分镜描述文本
        
        返回:
            dialogues: 提取的旁白/对白列表
            summary: 总结信息
        """
        # 使用正则表达式匹配所有分镜内容
        shot_pattern = r'<SHOT_(\d+)>([\s\S]*?)</SHOT_\1>'
        shot_matches = re.findall(shot_pattern, input_text)
        
        # 如果没有找到分镜标记，则直接在整个文本中查找旁白
        if not shot_matches:
            return self._extract_from_text(input_text)
        
        # 从每个分镜中提取旁白
        dialogues = []
        for shot_num, content in shot_matches:
            dialogue = self._extract_from_text(content)
            if dialogue[0]:  # 如果提取到了旁白
                dialogues.extend(dialogue[0])
        
        # 生成总结信息
        summary = f"一共提取了{len(dialogues)}条旁白/对白。"
        
        return (dialogues, summary)
    
    def _extract_from_text(self, text):
        """
        从文本中提取旁白/对白
        
        参数:
            text: 输入文本
        
        返回:
            dialogues: 提取的旁白/对白列表
            summary: 总结信息
        """
        # 使用正则表达式匹配旁白部分
        # 匹配"旁白："后面的方括号内容
        dialogue_pattern = r'旁白：\s*\[([^\[\]]*?)\]'
        dialogue_matches = re.findall(dialogue_pattern, text)
        
        # 如果没有找到旁白，尝试匹配其他可能的格式
        if not dialogue_matches:
            # 尝试匹配没有方括号的格式
            alt_pattern = r'旁白：\s*([^\[\]\n]*?)(?:\n|$)'
            dialogue_matches = re.findall(alt_pattern, text)
        
        dialogues = []
        for dialogue in dialogue_matches:
            # 去除开头和结尾的空白字符
            dialogue = dialogue.strip()
            if dialogue:  # 如果不是空字符串
                dialogues.append(dialogue)
        
        # 生成总结信息
        summary = f"提取了{len(dialogues)}条旁白/对白。"
        
        return (dialogues, summary)

# 节点映射字典，用于 ComfyUI 注册节点
NODE_CLASS_MAPPINGS = {
    "DialogueExtractor": DialogueExtractorNode
}

# 节点显示名称映射，用于在 UI 中显示友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "DialogueExtractor": "对白提取器 / Dialogue Extractor"
}