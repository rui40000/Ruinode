import os

def is_binary_file(path):
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
    except:
        return True
    return False

def try_decode(data):
    """
    尝试多种编码格式解码二进制数据 (对齐 fix_encoding.py)
    """
    # 1. UTF-8
    try:
        return data.decode("utf-8"), "utf-8"
    except:
        pass

    # 2. UTF-16 with BOM
    if data.startswith(b'\xff\xfe') or data.startswith(b'\xfe\xff'):
        try:
            return data.decode("utf-16"), "utf-16"
        except:
            pass

    # 3. GB18030 (covers GBK and more)
    try:
        return data.decode("gb18030"), "gb18030"
    except:
        pass
        
    # 4. Fallback
    return data.decode("utf-8", errors="ignore"), "fallback-ignore"

def clean_text(text, remove_emoji=True):
    """
    严格清洗文本 (对齐 fix_encoding.py)：
    - 删除控制字符 (保留换行和制表)
    - 删除 surrogate 区
    - 删除 emoji (默认强制删除 > 0xFFFF 的所有 4 字节字符)
    - 强制统一换行符为 Unix 风格 (\n)
    """
    cleaned = []
    for c in text:
        code = ord(c)

        # 删除控制字符（保留换行和制表）
        if 0 <= code < 32 and c not in ("\n", "\t"):
            continue

        # 删除 surrogate 区
        if 0xD800 <= code <= 0xDFFF:
            continue

        # 删除 emoji（默认删除所有 > 0xFFFF 字符）
        if remove_emoji and code > 0xFFFF:
            continue

        cleaned.append(c)

    result_text = "".join(cleaned)
    # 强制统一换行符为 Unix 风格 (\n)，防止 \r 导致打标工具报错
    result_text = result_text.replace("\r\n", "\n").replace("\r", "\n")
    return result_text

class UTF8ConverterNode:
    """
    UTF-8 编码转换与文本清洗节点 (工业级强力重构版)：
    完全对齐 fix_encoding.py 的逻辑，使用鲁棒的解码策略并执行极其严格的字符清洗。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "输入文本 / Input Text"
                }),
                "remove_emoji": ("BOOLEAN", {
                    "default": True,
                    "label_on": "是 / Yes",
                    "label_off": "否 / No",
                    "tooltip": "是否强制删除所有 4 字节字符（包含绝大多数 Emoji）"
                }),
            },
            "optional": {
                "input_file": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "文件路径 / File Path (优先读取 / Overrides input_text)"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filtered_text", "log")
    FUNCTION = "run"
    CATEGORY = "Rui-Node🐶/文本处理📝"

    def run(self, input_text, remove_emoji, input_file=""):
        log = []
        text_content = ""
        encoding_used = "none"

        # 1. 获取原始内容 (文件优先)
        if input_file and input_file.strip():
            if os.path.exists(input_file):
                log.append(f"正在读取文件: {input_file}")
                
                # 检查是否为二进制文件
                if is_binary_file(input_file):
                    log.append("警告: 检测到可能的二进制文件 (跳过)")
                    return ("", "\n".join(log))
                
                try:
                    with open(input_file, "rb") as f:
                        raw_data = f.read()
                    
                    text_content, encoding_used = try_decode(raw_data)
                    log.append(f"解码成功: 使用编码 {encoding_used}")
                    
                except Exception as e:
                    return ("", f"读取或解码文件时发生错误: {str(e)}")
            else:
                log.append(f"错误: 文件不存在 -> {input_file}")
                log.append("回退到使用 input_text")
                text_content = input_text
                encoding_used = "input_text"
        else:
            log.append("使用直接输入的文本")
            text_content = input_text
            encoding_used = "input_text"

        # 2. 清洗文本
        log.append("-" * 20)
        log.append("开始工业级强力清洗文本...")
        original_len = len(text_content)
        
        cleaned_content = clean_text(text_content, remove_emoji=remove_emoji)
        
        final_len = len(cleaned_content)
        removed_count = original_len - final_len
        
        # 3. 生成日志
        log.append(f"原始长度: {original_len}")
        log.append(f"输出长度: {final_len}")
        log.append(f"移除字符数: {removed_count}")
        
        if removed_count > 0:
            log.append(f"清洗详情: 已移除非法控制字符、代理对，并统一换行符为 \\n。")
            if remove_emoji:
                 log.append("已开启 remove_emoji: 删除了所有 4 字节字符。")
        else:
            log.append("结果: 文本已符合规范，无需修改。")

        return (cleaned_content, "\n".join(log))

# 节点映射字典
NODE_CLASS_MAPPINGS = {
    "UTF8Converter": UTF8ConverterNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "UTF8Converter": "转化为utf-8编码 / Convert to UTF-8"
}
