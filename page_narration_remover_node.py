class PageNarrationRemoverNode:
    """
    页面旁白删除器：删除文本中单独一行的页面旁白内容，如：
    页面旁白："……"
    
    输入：
        - input_text: 原始文本（支持多行）
    输出：
        - clean_text: 移除页面旁白行后的文本
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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("clean_text",)
    FUNCTION = "remove_narration"
    CATEGORY = "Rui-Node🐶/文本处理📝"

    def remove_narration(self, input_text: str):
        """
        删除文本中所有以“页面旁白：”或“页面旁白:”开头的整行（忽略前后空白）。
        保留其他内容，并以换行符连接返回。
        """
        if not isinstance(input_text, str):
            input_text = str(input_text)

        lines = input_text.splitlines()
        kept_lines = []
        for line in lines:
            stripped = line.strip()
            # 删除以“页面旁白：”或“页面旁白:”开头的整行
            if stripped.startswith("页面旁白：") or stripped.startswith("页面旁白:"):
                continue
            kept_lines.append(line)

        clean_text = "\n".join(kept_lines)
        return (clean_text,)


# 节点映射字典，用于 ComfyUI 注册节点
NODE_CLASS_MAPPINGS = {
    "PageNarrationRemover": PageNarrationRemoverNode
}

# 节点显示名称映射，用于在 UI 中显示友好名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "PageNarrationRemover": "页面旁白删除器 / Page Narration Remover"
}