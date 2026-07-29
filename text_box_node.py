# -*- coding: utf-8 -*-
"""
多行文本框（原样输出）节点
==========================
把输入的多行文本一字不动地输出为 STRING。

背景：WAS Node Suite 的「Text Multiline」节点会把 "#" 开头的行当注释删除，
还会做动态提示词/token 替换——喂 Markdown 文本时标题行（# 一级标题）会凭空
消失。本节点纯透传、零处理，专门用来安全地承载 Markdown / 代码等原样文本。
"""


class RuiTextBoxNode:
    """多行文本原样透传。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    # 关闭动态提示词处理，杜绝 {a|b}、通配符等被前端改写
                    "dynamicPrompts": False,
                    "tooltip": "多行文本，一字不动地原样输出。\n"
                               "不删注释行、不做动态提示词/通配符/token 替换。\n\n"
                               "用来替代 WAS 的「Text Multiline」——那个节点会把\n"
                               "「#」开头的行当注释删掉，喂 Markdown 时标题会凭空消失。\n"
                               "承载 Markdown、代码等格式敏感的文本请用本节点。"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "passthrough"
    CATEGORY = "Rui-Node🐶/文本处理📝"

    def passthrough(self, text):
        return (text,)


NODE_CLASS_MAPPINGS = {
    "RuiTextBox": RuiTextBoxNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiTextBox": "多行文本框(原样输出) / Text Box (Raw)",
}
