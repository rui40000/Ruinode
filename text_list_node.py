import torch
import json

class TextListNode:
    """
    文本列表制作器节点：将多个独立文本组织成列表形式输出
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            },
            "optional": {
                "text2": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "text3": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "text4": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "text5": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("text_list", "summary")
    FUNCTION = "create_text_list"
    CATEGORY = "Rui-Node🐶/文本处理📝"

    def create_text_list(self, text1, text2="", text3="", text4="", text5=""):
        """
        将多个文本组织成列表形式输出
        
        参数:
            text1-5: 输入的文本
        
        返回:
            text_list: 文本列表
            summary: 总结信息
        """
        # 收集所有非空文本
        text_list = []
        if text1.strip():
            text_list.append(text1.strip())
        if text2.strip():
            text_list.append(text2.strip())
        if text3.strip():
            text_list.append(text3.strip())
        if text4.strip():
            text_list.append(text4.strip())
        if text5.strip():
            text_list.append(text5.strip())
        
        # 生成总结信息
        summary = f"一共组织了{len(text_list)}段文本。"
        
        # 直接返回Python列表，不转换为字符串
        return (text_list, summary)

# 节点映射字典，用于 ComfyUI 注册节点
NODE_CLASS_MAPPINGS = {
    "TextList": TextListNode
}

# 节点显示名称映射，用于在 UI 中显示友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextList": "文本列表制作器 / Text List Creator"
}