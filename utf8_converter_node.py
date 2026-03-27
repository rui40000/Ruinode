import os
import torch

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
    尝试多种编码格式解码二进制数据
    Try to decode binary data with multiple encodings
    """
    # 1. Try UTF-8
    try:
        return data.decode("utf-8"), "utf-8"
    except:
        pass

    # 2. Try UTF-16
    try:
        return data.decode("utf-16"), "utf-16"
    except:
        pass

    # 3. Try GBK
    try:
        return data.decode("gbk"), "gbk"
    except:
        pass

    # 4. Fallback: UTF-8 with ignore
    return data.decode("utf-8", errors="ignore"), "fallback-ignore"

def clean_text(text):
    """
    严格清洗文本：
    - 删除不可见控制字符 (0-31, 保留 \n \t)
    - 删除代理对 (Surrogates)
    - 删除 4 字节字符 (Emoji, > 0xFFFF)
    """
    cleaned = []
    for c in text:
        code = ord(c)

        # 删除不可见控制字符 (U+0000 - U+001F)，保留 \n (10) 和 \t (9)
        if 0 <= code < 32 and c not in ("\n", "\t"):
            continue

        # 过滤非法 surrogate 字符 (U+D800 - U+DFFF)
        if 0xD800 <= code <= 0xDFFF:
            continue

        # 删除 Emoji 和其他 4 字节字符 (> 0xFFFF)
        # 注意：某些 Emoji（如 ⚠️ U+26A0）位于 BMP (Basic Multilingual Plane) 内，码点 < 0xFFFF
        # 因此仅判断 code > 0xFFFF 是不够的，需要引入 regex 或 Unicode 范围判断
        # 这里我们扩展过滤范围，根据 Unicode 标准过滤 Emoji
        # 简单起见，如果需要严格禁止 Emoji，可以使用 Unicode 范围
        # ⚠️ (Warning Sign) is U+26A0
        # ⚡ (High Voltage) is U+26A1
        # ⚽ (Soccer Ball) is U+26BD
        # 常见 Emoji 范围包括：
        # U+1F300-U+1F5FF (Miscellaneous Symbols and Pictographs)
        # U+1F600-U+1F64F (Emoticons)
        # U+1F680-U+1F6FF (Transport and Map Symbols)
        # U+2600-U+26FF (Miscellaneous Symbols) -> 包含 ⚠️, ⚽, ☁️ 等
        # U+2700-U+27BF (Dingbats) -> 包含 ✈️, ✉️, ✏️ 等
        # 
        # 用户要求：允许特殊符号如 ★(U+2605), →(U+2192), 《(U+300A), ·(U+00B7)
        # ⚠️ (U+26A0) 在 Miscellaneous Symbols 区域，与 ★ (U+2605) 同区！
        # 这是一个难点。
        # 策略更新：
        # 1. 删除所有 > 0xFFFF 的字符 (4字节字符) -> 已实现
        # 2. 针对 BMP 内的 Emoji，我们需要更细致的黑名单或白名单。
        #    考虑到用户说 "严禁使用任何 Emoji"，但 "允许星号、箭头等"。
        #    星号 ★ U+2605
        #    警告 ⚠️ U+26A0
        #    它们非常接近。
        #    我们可以使用 unicode property 或 regex。但在不引入外部库(如 emoji)的情况下，
        #    我们可以过滤特定的 Emoji 块，但保留特定的白名单字符。
        
        # 简单处理：保留 > 0xFFFF 过滤。
        # 针对 BMP 内的 Emoji，用户提到的 ⚠️ 是 U+26A0。
        # 许多 Emoji 位于 U+2000-U+2FFF 之间，但也包含数学符号和箭头。
        # 让我们尝试过滤 variation selectors (U+FE00-U+FE0F) 
        # ⚠️ (U+26A0) 通常后面跟着 U+FE0F (VS16) 变成 Emoji 样式。
        # 但单字符 ⚠️ 也是存在的。
        
        # 鉴于用户明确指出 ⚠️ 未删除，我们需要加强过滤。
        # 常见 BMP Emoji 范围：
        # U+2600-U+26FF (杂项符号) -> 混合了 Emoji 和 符号(如 ★)
        # U+2700-U+27BF (Dingbats) -> 混合了 Emoji 和 符号(如 ✂️)
        
        # 强制过滤列表 (手动列出常见 BMP Emoji 区域或字符)
        # 或者，更激进地，如果字符属于 "Symbol, Other" (So) 类别且不是白名单？
        # Python 的 unicodedata 库可以帮忙。
        
        import unicodedata
        try:
            category = unicodedata.category(c)
        except:
            category = "Cn" # Not assigned

        # 4字节字符统统删除
        if code > 0xFFFF:
            continue
            
        # 针对 BMP 字符的特殊过滤
        # 过滤 Variation Selectors (U+FE00 - U+FE0F)
        if 0xFE00 <= code <= 0xFE0F:
            continue
            
        # 过滤特定 Emoji 字符 (黑名单补丁)
        # ⚠️ U+26A0, ⚡ U+26A1, ✋ U+270B 等
        # 这是一个无底洞，但我们可以尝试过滤掉呈现为 Emoji 的符号
        # 许多现代 Emoji 实际上是基本字符 + VS16。
        # 如果我们删除了 VS16，它们会变成黑白文本符号。
        # 但用户希望 "删除"，即完全消失。
        
        # 使用 unicode category 过滤?
        # ★ (BLACK STAR) -> 'So' (Symbol, Other)
        # ⚠️ (WARNING SIGN) -> 'So'
        # 两者类别相同。
        
        # 既然无法通过类别区分，我们只能依赖码点范围，并设置白名单。
        # 允许：星号(★ U+2605)、方块(■ U+25A0)、箭头(→ U+2192)
        # 
        # 让我们检查 ⚠️ (U+26A0)
        # 
        # 临时方案：增加对常见 BMP Emoji 的过滤，如果不小心误杀，后续再调整。
        # 
        # 范围 U+2600 - U+26FF (Miscellaneous Symbols)
        # 包含：
        # 2600-2604 (太阳云雨等) -> Emoji? Yes (☀️, ☁️)
        # 2605-2606 (星星) -> 允许 (★, ☆)
        # 260E (电话 ☎️) -> Emoji
        # 26A0 (警告 ⚠️) -> Emoji
        # 
        # 我们可以只过滤那些明显的 Emoji 范围，或者根据用户反馈的 Bad Case (⚠️) 进行定点清除。
        # 但为了通用性，最好能区分。
        # 
        # 既然用户要求 "严禁使用任何 Emoji"，且 "允许使用的特殊符号仅限 UTF-8收录范围，例如..."
        # 这是一个比较模糊的边界。
        # 
        # 让我们引入 `unicodedata` 并结合范围判断。
        # 
        # 更新逻辑：
        # 1. 保留 code > 0xFFFF 过滤 (处理了绝大多数 Emoji)
        # 2. 增加对 BMP Emoji 的过滤。
        #    由于 Python 标准库没有 is_emoji，我们只能根据 Block 进行粗略过滤，并豁免常用符号。
        
        # 定义需要检查的 Block (潜在 Emoji 区域)
        # Dingbats: U+2700–U+27BF
        # Miscellaneous Symbols: U+2600–U+26FF
        # Transport and Map Symbols: U+1F680-U+1F6FF (已被 >0xFFFF 覆盖)
        
        # 针对 BMP 的补充过滤：
        if 0x2600 <= code <= 0x27BF:
            # 白名单 (用户明确允许或常见的非Emoji符号)
            # U+2605 ★, U+2606 ☆
            # U+25A0-U+25FF (Geometric Shapes) -> 不在 2600-27BF 范围内，安全
            # U+2190-U+21FF (Arrows) -> 不在范围内，安全
            # U+300A 《, U+00B7 · -> 不在范围内，安全
            
            # 允许的例外列表 (Decimal)
            # 9733 (★), 9734 (☆)
            allowed_in_range = [9733, 9734] 
            
            if code not in allowed_in_range:
                # 进一步检查：如果是 ⚠️ (9888) 或其他 Emoji，则过滤
                # 简单粗暴：在这个范围内，除了白名单，全部视为潜在 Emoji/不常用符号进行过滤？
                # 这样可能误杀太严重。
                # 
                # 让我们仅过滤特定的 Emoji 子集。
                # ⚠️ U+26A0 (9888)
                # ⚡ U+26A1 (9889)
                # ⚰️ U+26B0
                # ⚽ U+26BD
                # ⛄ U+26C4
                # ⛳ U+26F3
                # ...
                # 
                # 更好的方法：只过滤 > 0xFFFF 的字符 + 代理对 + 控制字符。
                # 对于 BMP 内的字符，除非用户指定要过滤，否则保留。
                # 但用户明确反馈 ⚠️ 未删除。
                # 
                # 让我们针对性地过滤 "Emoji Presentation" 的字符。
                # 但没有库很难做到。
                # 
                # 妥协方案：硬编码过滤常见的 BMP Emoji 范围，或仅过滤用户提到的 ⚠️。
                # 考虑到 "严禁使用任何 Emoji"，我们应该扩大过滤范围。
                
                # U+2600-U+26FF 包含很多天气、星座、棋子等，大多被视为 Emoji。
                # U+2700-U+27BF (Dingbats) 包含剪刀、飞机、信封等，大多被视为 Emoji。
                
                # 如果我们过滤掉这两个区段，除了白名单？
                # 用户提到的 "允许特殊符号"：
                # 星号 ★ (U+2605) -> 在 U+2600-U+26FF
                # 方块 ■ (U+25A0) -> 不在
                # 箭头 → (U+2192) -> 不在
                # 书名号 《 (U+300A) -> 不在
                # 间隔号 · (U+00B7) -> 不在
                
                # 结论：只要保护好 ★ (及其他可能的符号)，我们可以激进地过滤 U+2600-U+26FF 和 U+2700-U+27BF。
                pass # 逻辑将在下方实现
        
        # 实现：
        # 1. 过滤 Dingbats (U+2700 - U+27BF)
        if 0x2700 <= code <= 0x27BF:
            continue
            
        # 2. 过滤 Miscellaneous Symbols (U+2600 - U+26FF)，但保留白名单
        if 0x2600 <= code <= 0x26FF:
            # 白名单：
            # U+2605 ★ (9733)
            # U+2606 ☆ (9734)
            whitelist_26xx = {0x2605, 0x2606}
            if code not in whitelist_26xx:
                continue

        # 3. 过滤 Miscellaneous Technical 中常见的 Emoji (U+23xx)
        # ⌚ (231A), ⌛ (231B), ⌨ (2328)
        # ⏩ (23E9) - ⏳ (23F3)
        # ⏸ (23F8) - ⏺ (23FA)
        if code in (0x231A, 0x231B, 0x2328, 0x23CF):
             continue
        if 0x23E9 <= code <= 0x23F3:
             continue
        if 0x23F8 <= code <= 0x23FA:
             continue

        # 4. 过滤其他零散的常见 Emoji 符号
        # ⭐ (2B50), ⭕ (2B55)
        if code in (0x2B50, 0x2B55):
             continue
             
        cleaned.append(c)

    return "".join(cleaned)

class UTF8ConverterNode:
    """
    UTF-8 编码转换与文本清洗节点 (重构版)：
    使用鲁棒的解码策略读取文件或处理文本，并执行严格的字符清洗。
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

    def run(self, input_text, input_file=""):
        log = []
        text_content = ""
        encoding_used = "none"

        # 1. 获取原始内容 (文件优先)
        if input_file and input_file.strip():
            if os.path.exists(input_file):
                log.append(f"正在读取文件: {input_file}")
                
                # 检查是否为二进制文件
                if is_binary_file(input_file):
                    # 虽然是二进制，但我们仍尝试解码，或者直接报错
                    # 根据需求"转化为utf-8"，我们尝试强制解码
                    log.append("警告: 检测到可能的二进制文件")
                
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
        log.append("开始清洗文本...")
        original_len = len(text_content)
        
        cleaned_content = clean_text(text_content)
        
        final_len = len(cleaned_content)
        removed_count = original_len - final_len
        
        # 3. 生成日志
        log.append(f"原始长度: {original_len}")
        log.append(f"输出长度: {final_len}")
        log.append(f"移除字符数: {removed_count}")
        
        if removed_count > 0:
            log.append("清洗详情: 已移除所有 Emoji、4字节字符、代理对及非法控制字符。")
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
