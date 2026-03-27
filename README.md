# Rui-Node🐶 - ComfyUI 图像处理节点集

Rui-Node🐶 是一个功能丰富的 ComfyUI 节点集合，提供图像处理、文本处理、AI 模型集成和遮罩处理等多种功能。

## 📦 安装方法

1. 将此文件夹复制到 ComfyUI 的 `custom_nodes` 目录中
2. 安装依赖：`pip install -r requirements.txt`
3. 重启 ComfyUI

## 📋 节点目录

### 🎨 图像调节类
- [调整饱和度 / Saturation Adjustment](#1-调整饱和度--saturation-adjustment)
- [图像翻转 / Image Flip](#2-图像翻转--image-flip)
- [颜色匹配器 / Color Matcher](#13-颜色匹配器--color-matcher)
- [素材拆分 / Sprite Splitter](#14-素材拆分--sprite-splitter)
- [素材拆分(带透明通道) / Sprite Splitter RGBA](#15-素材拆分带透明通道--sprite-splitter-rgba)

### 📁 文件存储与加载类
- [按路径加载图像 / Load Image By Path](#3-按路径加载图像--load-image-by-path)

### 🤖 AI模型类
- [千问编辑图像生成 / Qwen Edit Image Generation](#4-千问编辑图像生成--qwen-edit-image-generation)

### 📝 文本处理类
- [镜头分词器 / Shot Splitter](#5-镜头分词器--shot-splitter)
- [对白提取器 / Dialogue Extractor](#6-对白提取器--dialogue-extractor)
- [页面旁白删除器 / Page Narration Remover](#7-页面旁白删除器--page-narration-remover)
- [文本列表制作器 / Text List Creator](#8-文本列表制作器--text-list-creator)
- [转化为utf-8编码 / Convert to UTF-8](#11-转化为utf-8编码--convert-to-utf-8)

### 🎭 遮罩处理类
- [遮罩筛选 / Mask Selector](#9-遮罩筛选--mask-selector)
- [遮罩预览 / Mask Preview](#10-遮罩预览--mask-preview)

---

## 📖 节点详细说明

### 1. 调整饱和度 / Saturation Adjustment

**分类**: `Rui-Node🐶/图像调节🎨`

**功能描述**:  
调整图像的色彩饱和度，可以创建黑白图像或增强色彩鲜艳度。

**输入参数**:
- `image` (IMAGE): 输入图像
- `saturation` (FLOAT): 饱和度调整系数
  - 默认值: 1.0
  - 范围: 0.0 ~ 5.0
  - 步长: 0.1
  - 说明:
    - 0.0 = 完全无饱和度（黑白图像）
    - 1.0 = 原始饱和度（不变）
    - >1.0 = 增加饱和度

**输出**:
- `IMAGE`: 调整后的图像

**使用场景**:
- 将彩色图像转换为黑白
- 增强图像色彩表现力
- 降低过于鲜艳的色彩

---

### 2. 图像翻转 / Image Flip

**分类**: `Rui-Node🐶/图像调节🎨`

**功能描述**:  
对图像进行水平或垂直翻转操作。

**输入参数**:
- `image` (IMAGE): 输入图像
- `flip_direction` (选择): 翻转方向
  - 选项: "水平" 或 "垂直"
  - 默认值: "水平"

**输出**:
- `IMAGE`: 翻转后的图像

**使用场景**:
- 镜像翻转图像
- 创建对称效果
- 调整图像方向

---

### 3. 按路径加载图像 / Load Image By Path

**分类**: `Rui-Node🐶/文件存储与加载📁`

**功能描述**:  
从指定的文件路径加载图像文件，支持绝对路径输入。

**输入参数**:
- `image_path` (STRING): 图像文件的完整路径
  - 默认值: "E:\\ComfyUIModels\\input\\10\\1.png"
  - 支持格式: PNG、JPG、JPEG 等常见图像格式

**输出**:
- `IMAGE`: 加载的图像

**特殊处理**:
- 如果文件不存在，返回 512x512 的黑色默认图像
- 自动将非 RGB 图像转换为 RGB 模式

**使用场景**:
- 从外部路径加载特定图像
- 批量处理指定目录的图像
- 加载非 ComfyUI 默认输入目录的图像

---

### 4. 千问编辑图像生成 / Qwen Edit Image Generation

**分类**: `Rui-Node🐶/AI模型🤖`

**功能描述**:  
使用阿里云千问（Qwen）编辑模型 API 进行 AI 图像生成，支持多种控制模式。

**输入参数**:
- `image1` ~ `image4` (IMAGE): 最多 4 张输入图像作为参考
- `api_key` (STRING): 阿里云 API 密钥
- `base_url` (STRING): API 基础 URL
  - 默认值: "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image-generation/generation"
- `seed` (INT): 随机种子
  - 默认值: -1（随机生成）
  - 范围: -1 ~ 2147483647
- `control_mode` (选择): 控制模式
  - 选项: reference, sketch, scribble, pose, canny, depth, hed, mlsd, normal, seg
  - 默认值: "reference"
- `width` (INT): 输出图像宽度
  - 默认值: 1024
  - 范围: 512 ~ 2048
  - 步长: 8
- `height` (INT): 输出图像高度
  - 默认值: 1024
  - 范围: 512 ~ 2048
  - 步长: 8

**输出**:
- `IMAGE`: AI 生成的图像

**使用场景**:
- AI 辅助图像创作
- 基于参考图生成新图像
- 多模态图像控制生成

---

### 5. 镜头分词器 / Shot Splitter

**分类**: `Rui-Node🐶/文本处理📝`

**功能描述**:  
将包含多个分镜描述的脚本文本拆分成独立的分镜列表，支持按范围筛选导出。

**输入参数**:
- `input_text` (STRING): 输入的多分镜描述脚本（多行文本）
  - 格式要求: 使用 `<SHOT_XXX>...</SHOT_XXX>` 标签包裹每个分镜
- `start_shot_num` (INT, 可选): 开始导出的分镜编号
  - 默认值: 0（从第一个开始）
  - 范围: 0 ~ 100
- `shot_count` (INT, 可选): 导出的分镜数量
  - 默认值: 0（导出全部）
  - 范围: 0 ~ 100

**输出**:
- `shot_descriptions` (LIST): 拆分后的分镜描述列表
- `summary` (STRING): 总结信息

**文本格式示例**:
```
<SHOT_1>
第一个镜头的描述内容
</SHOT_1>
<SHOT_2>
第二个镜头的描述内容
</SHOT_2>
```

**使用场景**:
- 分镜脚本拆分
- 批量处理分镜描述
- 选择性导出特定范围的分镜

---

### 6. 对白提取器 / Dialogue Extractor

**分类**: `Rui-Node🐶/文本处理📝`

**功能描述**:  
从分镜描述文本中自动提取旁白/对白内容。

**输入参数**:
- `input_text` (STRING): 输入的分镜描述文本（多行文本）

**输出**:
- `dialogues` (LIST): 提取的旁白/对白列表
- `summary` (STRING): 总结信息

**识别模式**:
- 支持格式 1: `旁白：[对白内容]`
- 支持格式 2: `旁白：对白内容`
- 自动识别 `<SHOT_XXX>` 标签中的旁白

**使用场景**:
- 从分镜脚本中提取对白
- 批量收集旁白文本
- 准备配音文本

---

### 7. 页面旁白删除器 / Page Narration Remover

**分类**: `Rui-Node🐶/文本处理📝`

**功能描述**:  
删除文本中所有以"页面旁白："或"页面旁白:"开头的整行内容。

**输入参数**:
- `input_text` (STRING): 原始文本（多行文本）

**输出**:
- `clean_text` (STRING): 移除页面旁白行后的文本

**处理规则**:
- 自动识别并删除以"页面旁白："或"页面旁白:"开头的行
- 忽略行首行尾的空白字符
- 保留其他所有内容

**使用场景**:
- 清理脚本中的页面旁白
- 文本预处理
- 提取纯净对白内容

---

### 8. 文本列表制作器 / Text List Creator

**分类**: `Rui-Node🐶/文本处理📝`

**功能描述**:  
将多个独立的文本段落组织成列表形式输出。

**输入参数**:
- `text1` (STRING, 必需): 第一段文本（多行文本）
- `text2` ~ `text5` (STRING, 可选): 第 2~5 段文本（多行文本）

**输出**:
- `text_list` (LIST): 文本列表（Python 列表格式）
- `summary` (STRING): 总结信息

**处理规则**:
- 自动过滤空文本
- 去除每段文本首尾的空白字符
- 保留内部段落结构

**使用场景**:
- 组织多段文本为列表
- 批量文本处理准备
- 文本分组管理

---

### 9. 遮罩筛选 / Mask Selector

**分类**: `Rui-Node🐶/遮罩处理🎭`

**功能描述**:  
对输入的多个遮罩进行排序并选择特定遮罩，同时输出剩余遮罩的合并结果。

**输入参数**:
- `masks` (MASK): 输入的遮罩（可包含多个遮罩）
- `sort_method` (选择): 排序方法
  - 选项:
    - "按面积排序 / By Area": 按遮罩面积从大到小排序
    - "从左到右 / Left to Right": 按遮罩质心 X 坐标升序排序
    - "从上到下 / Top to Bottom": 按遮罩质心 Y 坐标升序排序
  - 默认值: "按面积排序 / By Area"
- `index` (INT): 选择的遮罩编号（1-based 索引）
  - 默认值: 1
  - 最小值: 1
  - 说明: 如果超出范围会自动夹取到有效范围

**输出**:
- `选中遮罩 / Selected` (MASK): 选中的单个遮罩
- `剩余遮罩 / Remaining` (MASK): 其他遮罩的合并结果
- `信息 / Info` (STRING): JSON 格式的详细信息
  - `total_masks`: 遮罩总数
  - `selected_index`: 选中编号
  - `sort_method`: 排序方式
  - `selected_area`: 选中遮罩的像素面积
  - `selected_center`: 选中遮罩的质心坐标 [x, y]
  - `index_clamped`: 编号是否越界被修正

**使用场景**:
- 从多个检测结果中选择特定目标
- 分离主体和背景遮罩
- 基于大小或位置筛选遮罩

---

### 10. 遮罩预览 / Mask Preview

**分类**: `Rui-Node🐶/遮罩处理🎭`

**功能描述**:  
将遮罩以半透明彩色形式叠加显示在图像上，方便直观查看遮罩覆盖区域。节点自带预览功能，同时输出合成后的图像。

**输入参数**:
- `image` (IMAGE): 作为底图的原始图像
- `mask` (MASK): 需要可视化的遮罩
- `mask_color` (选择): 遮罩显示颜色
  - 默认值: 红色 / Red
  - 选项: 红色、绿色、蓝色、黄色、青色、品红、白色
- `opacity` (FLOAT, 可选): 不透明度
  - 默认值: 0.5
  - 范围: 0.0 ~ 1.0
  - 步长: 0.05

**输出**:
- `图像 / Image` (IMAGE): 合成了半透明彩色遮罩的图像

**特性**:
- 自动处理遮罩与图像的尺寸差异
- 节点界面直接显示预览效果
- 支持批量处理
- 7种预设颜色可选
- 可调节不透明度

**使用场景**:
- 检查分割结果的准确性
- 调试遮罩处理流程
- 多遮罩对比（使用不同颜色）
- 制作遮罩可视化图

---

### 11. 转化为utf-8编码 / Convert to UTF-8

**分类**: `Rui-Node🐶/文本处理📝`

**功能描述**:  
删除输入字符串中所有非 UTF-8 编码字符（如孤立的代理对），确保输出的字符串符合 UTF-8 编码规范。

**输入参数**:
- `input_text` (STRING): 需要处理的原始字符串（支持多行）

**输出**:
- `filtered_text` (STRING): 过滤后的符合 UTF-8 规范的字符串
- `log` (STRING): 处理日志，包含移除字符的详细信息和统计总结

**使用场景**:
- 清理可能包含非法字符的文本数据
- 确保文本在保存或传输时的编码安全性
- 调试文本编码问题

---

### 12. OpenAI API 连接 / OpenAI API Connector

**分类**: `Rui-Node🐶/AI模型🤖`

**功能描述**:  
连接 OpenAI 或兼容 API（如 DeepSeek、Moonshot 等），进行文本生成或多模态图像理解。

**输入参数**:
- `api_url` (STRING): API 接口地址
  - 默认值: "https://api.openai.com/v1/chat/completions"
- `api_key` (STRING): API 密钥
- `model` (STRING): 模型名称
  - 默认值: "gpt-4o"
- `system_prompt` (STRING): 系统提示词
- `user_prompt` (STRING): 用户提示词
- `seed` (INT): 随机种子，用于控制生成的随机性
- `image` (IMAGE, 可选): 输入图像（用于多模态模型）

**输出**:
- `text` (STRING): 模型生成的文本内容

**使用场景**:
- 调用 LLM 进行文本生成
- 使用 Vision 模型进行图像理解
- 连接本地或第三方兼容 OpenAI 协议的 API

---

### 13. 颜色匹配器 / Color Matcher

**分类**: `Rui-Node🐶/图像调节🎨`

**功能描述**:  
将目标图像的颜色分布匹配到参考图像的颜色分布，支持多种匹配算法和混合调节。

**输入参数**:
- `reference_image` (IMAGE): 作为颜色参考的图像
- `moving_image` (IMAGE): 需要改变颜色的目标图像
- `match_method` (选择): 匹配算法
  - 选项: "histogram" (直方图匹配), "mean_std" (均值标准差匹配), "none" (无匹配)
  - 默认值: "histogram"
- `blend_factor` (FLOAT): 混合系数
  - 默认值: 1.0
  - 范围: 0.0 ~ 1.0
  - 步长: 0.01
  - 说明: 控制原图和匹配后图像的混合比例，1.0为完全使用匹配后图像

**输出**:
- `颜色匹配后图像` (IMAGE): 颜色调整后的图像
- `匹配信息` (STRING): 记录了使用的匹配方式以及混合系数的日志信息

**使用场景**:
- 统一多张图像的色调风格
- 将素材无缝融合进背景
- 图像色彩风格迁移

---

### 14. 素材拆分 / Sprite Splitter

**分类**: `Rui-Node🐶/图像调节🎨`

**功能描述**:  
从白色/浅色背景的合图（Sprite Sheet）中自动拆分出每个独立的美术元素，通过连通区域检测进行裁剪，并将每个独立元素作为图像列表输出。

**输入参数**:
- `图像` (IMAGE): 输入的带有透明通道的合图图像（RGBA格式）
- `最小面积过滤（像素数）` (INT): 最小面积过滤
  - 默认值: 100
  - 范围: 1 ~ 50000
  - 说明: 面积小于此值（像素数）的连通区域将被过滤，避免拆分出噪点碎片。
- `裁剪边距` (INT): 裁剪边距
  - 默认值: 2
  - 范围: 0 ~ 50
  - 说明: 每个元素裁剪时在包围盒外额外保留的像素边距。
- `排序方式` (选择): 排序方式
  - 选项: "从左到右-从上到下", "从上到下-从左到右", "面积从大到小", "面积从小到大"
  - 默认值: "从左到右-从上到下"

**输出**:
- `图像列表` (IMAGE): 拆分后的多张图像列表，透明区域会用白色填充输出。

**使用场景**:
- 游戏素材合图切分
- 批量图标提取
- 白底素材自动裁剪

---

### 15. 素材拆分(带透明通道) / Sprite Splitter RGBA

**分类**: `Rui-Node🐶/图像调节🎨`

**功能描述**:  
与标准素材拆分节点功能相同，但保留并额外输出 Alpha 透明通道，适用于需要透明背景的美术素材提取。

**输入参数**:
- 输入参数与 [素材拆分 / Sprite Splitter](#14-素材拆分--sprite-splitter) 完全一致。

**输出**:
- `图像列表` (IMAGE): 拆分后的多张 RGB 图像列表
- `遮罩列表` (MASK): 对应的多张 Alpha 透明通道遮罩列表，1.0代表不透明，0.0代表透明

**使用场景**:
- 提取带透明背景的游戏角色、道具素材
- 搭配 `JoinImageWithAlpha` 等节点生成透明 PNG 图像

---

## 🔧 依赖库

主要依赖库包括：
- `torch`: PyTorch 深度学习框架
- `numpy`: 数值计算
- `Pillow (PIL)`: 图像处理
- `requests`: HTTP 请求（用于 API 调用）

完整依赖请查看 `requirements.txt`

## 📝 注意事项

1. 所有节点都兼容 ComfyUI 的标准图像处理流程
2. 图像格式统一为 BHWC（批次、高度、宽度、通道）
3. 图像值范围为 0.0 ~ 1.0 的浮点数
4. 使用 AI 模型节点需要配置有效的 API 密钥
5. 文本处理节点支持多行文本输入
6. 所有节点名称采用中英双语显示
7. 遮罩处理节点自动处理尺寸不匹配问题

## 🐕 关于 Rui-Node🐶

Rui-Node🐶 致力于为 ComfyUI 用户提供实用、高效的节点工具集。🐶 是我们的项目标志，代表着忠诚、友好和可靠。

## 📄 许可证

本项目遵循开源协议，欢迎使用和贡献。

---

**Happy Creating with Rui-Node🐶!** 🎨✨
