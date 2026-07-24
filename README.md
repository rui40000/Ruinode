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
- [加载图像(带文件名) / Load Image With Name](#16-加载图像带文件名--load-image-with-name)

### 🤖 AI模型类
- [千问编辑图像生成 / Qwen Edit Image Generation](#4-千问编辑图像生成--qwen-edit-image-generation)
- [SDMatte 精细抠图 / SDMatte Interactive Matting](#17-sdmatte-精细抠图--sdmatte-interactive-matting)
- [ZenMux API 连接 / ZenMux API Connector](#18-zenmux-api-连接--zenmux-api-connector)

### 📝 文本处理类
- [镜头分词器 / Shot Splitter](#5-镜头分词器--shot-splitter)
- [对白提取器 / Dialogue Extractor](#6-对白提取器--dialogue-extractor)
- [页面旁白删除器 / Page Narration Remover](#7-页面旁白删除器--page-narration-remover)
- [文本列表制作器 / Text List Creator](#8-文本列表制作器--text-list-creator)
- [转化为utf-8编码 / Convert to UTF-8](#11-转化为utf-8编码--convert-to-utf-8)
- [Markdown转图片 / Markdown To Image](#19-markdown转图片--markdown-to-image)
- [多行文本框(原样输出) / Text Box (Raw)](#20-多行文本框原样输出--text-box-raw)

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
连接 OpenAI 或兼容 API（如 DeepSeek、Moonshot 等），进行文本生成或多模态图像理解，支持最多 6 张图像同时输入。

**输入参数**:
- `api_url` (STRING): API 接口地址
  - 默认值: "https://api.openai.com/v1/chat/completions"
- `api_key` (STRING): API 密钥
- `model` (STRING): 模型名称
  - 默认值: "gpt-4o"
- `system_prompt` (STRING): 系统提示词
- `user_prompt` (STRING): 用户提示词
- `seed` (INT): 随机种子，用于控制生成的随机性
- `image_1` ~ `image_6` (IMAGE, 可选): 最多 6 张输入图像
  - 说明: 用户有几张图就连接几个输入口，无需手动 Batch
  - 规则: 节点内部会自动逐张处理每个输入图像，分别编码后发送到 API
  - 优势: 不要求所有图像尺寸一致，512×512 和 511×768 之类的混合输入也可直接使用
- `temperature` (FLOAT, 可选): 采样温度
  - 默认值: 0.3
  - 范围: 0.0 ~ 2.0
- `max_tokens` (INT, 可选): 最大输出 token 数
  - 默认值: 500
  - 范围: 1 ~ 8192
- `detail` (选择, 可选): 图像分析细节等级
  - 选项: low, high, auto
  - 默认值: auto
- `image_max_size` (INT, 可选): 单张图像最长边缩放上限
  - 默认值: 1024
  - 范围: 256 ~ 4096
  - 说明: 超过该尺寸的图像会在发送前按比例缩小，以减少 token 消耗与请求体积
- `proxy_url` (STRING, 可选): HTTP/HTTPS 代理地址
  - 示例: `http://127.0.0.1:7890`

**输出**:
- `text` (STRING): 模型生成的文本内容

**使用场景**:
- 调用 LLM 进行文本生成
- 使用 Vision 模型进行单图或多图联合理解
- 连接本地或第三方兼容 OpenAI 协议的 API
- 对多张参考图做综合分析、比对与总结

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
- `seed` (INT): 随机种子
  - 默认值: 0
  - 说明: 仅用于强制重新执行节点，不影响实际拆分结果。适用于线上部署时强制刷新缓存。

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

### 16. 加载图像(带文件名) / Load Image With Name

**分类**: `Rui-Node🐶/文件存储与加载📁`

**功能描述**:  
基础功能与 ComfyUI 原生的 "Load Image" 节点完全一致，支持从 ComfyUI 的 `input` 目录中选择图像，并支持拖拽上传。区别在于本节点额外提供了一个字符串输出端口，用于输出图像的文件名。

**输入参数**:
- `image` (下拉选择): 从 `input` 目录中选择图像文件，或通过按钮上传

**输出**:
- `IMAGE`: 图像数据
- `MASK`: 图像的 Alpha 通道遮罩
- `filename` (STRING): 图像的文件名（不包含后缀，例如上传了 `test_image.png`，则输出 `test_image`）

**使用场景**:
- 批量处理图像时，希望以原文件名保存处理后的结果
- 需要将当前图像的文件名作为提示词或其他参数传递给下游节点
- 建立更规范的自动化工作流

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

### 17. SDMatte 精细抠图 / SDMatte Interactive Matting

基于 [SDMatte](https://github.com/vivoCameraResearch/SDMatte)（vivo 相机研究院，ICCV 2025）的交互式抠图节点。
擅长发丝、绒毛、玻璃、烟雾等常规抠图模型处理不好的边缘。

包含两个节点：

| 节点 | 作用 |
|---|---|
| **SDMatte 加载器** | 载入权重，构建网络并常驻显存 |
| **SDMatte 精细抠图** | 用视觉提示（框/掩码/点）驱动模型输出 alpha |

#### 模型准备

把权重放到 `ComfyUI/models/SDMatte/` 下即可，两种格式任选其一：

- `SDMatte_plus.pth` — 官方发布，12.1GB，[LongfeiHuang/SDMatte](https://huggingface.co/LongfeiHuang/SDMatte)
- `SDMatte_plus.safetensors` — 社区转换，5.19GB，[1038lab/SDMatte](https://huggingface.co/1038lab/SDMatte)

> **这两个文件的模型权重逐比特完全相同**，不必纠结选哪个。
> 已实测比对全部 1316 个张量：键名、形状、精度（均为 F32）、数值全部一致，无一例外。
> 官方 pth 是 detectron2 的训练检查点，顶层为 `{"model", "trainer", "iteration"}`，
> 多出的约 6.9GB 是 `trainer` 里的优化器状态与梯度缩放器，推理不参与。
> 换用 pth **不会带来任何质量提升**。本节点两种格式都支持，读 pth 时只解析 `model` 段，
> 内存占用与 safetensors 相当。

**不需要下载 Stable Diffusion 2.1 的权重。** SDMatte 虽以 SD 2.1 为骨架，但官方推理配置
（`configs/SDMatte.py` 中 `load_weight=False`）只用配置文件搭出网络结构，全部权重随后由
SDMatte 检查点覆盖。官方 HuggingFace 仓库本身也只发布 `.pth` 加若干 `config.json`，
不含任何 SD 权重。所需配置已随本节点一起分发，开箱即用、无需联网。

#### 参数说明

**SDMatte 加载器**

| 参数 | 说明 |
|---|---|
| `ckpt_name` | `models/SDMatte/` 下的权重文件 |
| `precision` | `fp32`（默认，与官方测试配置一致）/ `fp16`（省显存，但 SD 2.1 的 VAE 半精度下易溢出） |
| `device` | `auto` / `cpu` |
| `attention_slicing` | 默认开启。1024 下显存峰值从约 **15.5GB 降到 9.1GB**，实测速度反而略快，输出差异仅 1e-6 量级 |

> 显存参考（fp32 @ 1024，实测于 RTX 5090）：开分片约 **9.1GB**，关分片约 **15.5GB**。
> 12GB 显存的卡请保持分片开启。

**SDMatte 精细抠图**

| 参数 | 说明 |
|---|---|
| `mask` | 指示抠哪个目标的提示掩码，**不必精确**，粗略覆盖主体即可 |
| `prompt_type` | 视觉提示类型，见下表 |
| `inference_size` | 默认 `1024`，与官方测试一致 |
| `is_transparent` | 玻璃、纱、烟雾等透明物体**务必打开** |
| `caption` | 目标物体的英文描述。**仅 `SDMatte.pth` 有效，`SDMatte_plus.pth` 请留空**，见下文 |
| `point_radius` | 仅 `point_mask` 生效。每个点晕开的高斯 sigma，默认 35 |
| `seed` | 仅 `point_mask` 生效（10 个点是随机取的） |

`prompt_type` 选择：

| 取值 | 含义 | 适用 |
|---|---|---|
| `bbox_mask` | 取掩码外接框作为提示 | **默认，官方测试脚本的主路径，通常最稳** |
| `mask` | 直接用掩码本身 | 已有较准的粗分割时 |
| `point_mask` | 在掩码内随机取 10 个点 | **仅 `SDMatte.pth` 支持**，见下文 |
| `auto_mask` | 不给定位信息 | 画面只有单一主体 |

#### ⚠ 两个权重的能力不同（实测）

官方 README 里，**SDMatte** 与 **SDMatte\***（即 `SDMatte_plus`）的训练集不同：
前者含 **RefMatte**（指代表达式抠图数据集，点提示与文本提示的来源），
后者用 **COCO-Matte** 替换了它。这导致 plus 版**不具备点提示与文本指代能力**：

| | `SDMatte.pth` | `SDMatte_plus.pth` |
|---|---|---|
| `bbox_mask` / `mask` / `auto_mask` | ✅ | ✅ |
| `point_mask` | ✅ MAD 0.0135 | ❌ **输出全黑**（max 仅 0.079） |
| `caption` 语义 | ✅ 填对小幅提升 | ❌ 无作用，填了反而更差 |

`caption` 实测（羊驼图，MAD 越低越好）：

| caption | `SDMatte` | `SDMatte_plus` |
|---|---|---|
| `""`（留空） | 0.01120 | **0.01135** ← 最好 |
| `"alpaca"`（语义正确） | **0.01072** ← 最好 | 0.01160 ← 最差 |
| `"tree"`（语义错误） | 0.01111 | 0.01119 |

在 `SDMatte` 上，语义正确的描述确实更准；在 `plus` 上语义完全失效甚至反向，
说明它只是给 cross-attention 注入了噪声扰动，并非在理解文本。

**结论**：用 `SDMatte_plus.pth` 时保持 `caption` 留空、`prompt_type` 用 `bbox_mask`；
想用点提示或文本指代，请换 `SDMatte.pth`。节点在 `point_mask` 输出接近全黑时会打印警告。

#### 典型接法

```
加载图像 ──────────────┬──> SDMatte 精细抠图 ──> alpha (MASK)
                       │         ▲                 └──> cutout (IMAGE)
任意分割节点 ──> mask ──┘         │
SDMatte 加载器 ───────────────────┘
```

`mask` 可以来自任何粗分割来源（SAM、rembg、手绘遮罩皆可）——SDMatte 的职责正是把粗糙边缘细化。

#### 实测数据

用官方效果图中的羊驼原图（绒毛边缘）跑本节点，与官方给出的 GT alpha 对比：

| 指标 | 数值 |
|---|---|
| MAD（平均绝对误差） | 0.0113 |
| MSE | 0.0026 |
| SAD | 0.807 千像素 |

（GT 取自官方效果图截图，含有损压缩与水印，故存在固有误差下限。）

各配置对输出的实际影响（透明玻璃杯，差异像素指偏差 > 0.05 的占比）：

| 对照项 | 平均差 | 差异像素占比 |
|---|---|---|
| `inference_size` 1024 vs 512 | 0.082 | 32.4% |
| `is_transparent` 关 vs 开 | 0.059 | 25.0% |
| 官方 `[F,T,F]` vs 误用 `[T,T,T]` 条件分配 | 0.026 | 18.8% |

结论：**分辨率影响最大，建议保持 1024**；抠透明物体时 `is_transparent` 必须打开。

#### 与 ComfyUI-SDMatte 的横向实测

同一张图、同一份权重、同一台机器，对跑 [ComfyUI-SDMatte](https://github.com/flybirdxx/ComfyUI-SDMatte)
与本节点，以官方公布的 alpha 为参照：

| 实现 | 配置 | MAD ↓ |
|---|---|---|
| **本节点** | 官方 `configs/SDMatte.py`，bbox 提示，fp32 | **0.0113** |
| ComfyUI-SDMatte | 默认（trimap 提示 + `mask_refine`） | 0.0884 |
| ComfyUI-SDMatte | 关闭 `mask_refine` | 0.0885 |

**相差 7.8 倍**，且其输出肉眼可见地发灰、边缘晕开。

主因是**视觉提示类型**：官方 `configs/SDMatte.py` 固定 `aux_input="bbox_mask"`，
而其 `aux_input_list` 只含 `point_mask` / `bbox_mask` / `mask` —— **trimap 从未作为视觉提示参与训练**。
ComfyUI-SDMatte 传 `aux_input="trimap"`，把模型推到了没训练过的输入模式上，
且该分支的 `trimap_coords` 恒为 `[0,0,1,1]`，定位信息全部丢失。
开不开它的 `mask_refine` 几乎不影响这一结论（0.0884 vs 0.0885），说明问题不在后处理。

#### 实现要点

若与其它 SDMatte 实现效果对不上，按影响从大到小排查：

1. **视觉提示类型**（影响最大）。必须用官方训练过的 `bbox_mask` / `mask` / `point_mask`，
   并传入真实的归一化坐标。用 trimap 当视觉提示是模型没见过的用法。

2. **UNet 配置来源**。SDMatte 在标准 SD 2.1 的 UNet 配置上额外定义了
   `bbox_time_embed_dim` / `point_embeddings_input_dim` / `bbox_embeddings_input_dim` 三个字段。
   误用原版 SD 2.1 的 `config.json` 会缺这些字段，只能猜默认值，猜错则相应权重被
   `strict=False` 静默丢弃。本节点直接分发官方配置，并在缺字段时**直接报错而非猜测**。

3. **transformers 版本**。官方权重用 transformers 4.x 保存，`CLIPTextModel` 内部裹了一层
   `text_model`；transformers 5.x 起该层被移除，导致 text_encoder 的 372 个权重键名对不上、
   被整体静默丢弃、停留在随机初始化。本节点会按当前环境自动增删该前缀。

4. **条件分配**。官方 `use_encoder_hidden_states_list=[False, True, False]` 决定 UNet
   下采样/中间/上采样三段各接收哪种条件，漏传会退化成 `[True, True, True]`。
   实测单独影响不大（羊驼 MAD 0.01135 → 0.01148），透明物体上更明显。

5. **权重对齐校验**。本节点在加载后校验键的完整性，一旦有权重未被覆盖或未被使用就**中止并报错**。
   这类问题不会让模型崩溃，只会让输出质量悄悄下降，是最难排查的一类，因此宁可停下也不放行。

6. 全程 fp32、1024 分辨率，且**不做任何启发式后处理**（不做阈值裁剪、对比度拉伸之类的"优化"），
   输出即模型原始 alpha。

---

### 18. ZenMux API 连接 / ZenMux API Connector

**分类**: `Rui-Node🐶/AI模型🤖`

**功能描述**:  
连接 [ZenMux](https://zenmux.ai) 聚合平台（OpenAI 兼容协议），一个节点即可调用其收录的**所有文本类模型**（Anthropic、OpenAI、Google、DeepSeek、Qwen 等 20 家厂商、130+ 模型）。支持文本生成与多模态图像理解（最多 6 张图）。

**特色功能**:
- **价格直接标在选项上**: 每个模型后缀形如 `[入$0.2/M 出$1.25/M]`，即输入/输出每百万 token 的美元价格，选型时一目了然
- **快速筛选**: 模型列表按「厂商/模型名」排序聚类，同厂商模型天然相邻；在下拉的搜索框输入厂商前缀（如 `qwen/`、`anthropic/`）即可只看该厂商的模型
- **离线可用的模型清单**: 模型与价格来自随包分发的 `zenmux/models_snapshot.json`；价格有变动时运行 `python zenmux/build_snapshot.py` 即可重新拉取更新
- **旧工作流兼容**: 价格快照更新后，旧工作流里保存的带旧价格标签仍能正确解析出模型 id，不会失效
- **单次消耗统计**: `usage_stats` 输出本次运行的 token 用量、输出字数与费用换算（按快照单价计算，汇率可调），格式：

  ```
  token消耗，输入：1234，输出：567
  输出文字数量：328
  模型类型：openai/gpt-5.4-nano [入$0.2/M 出$1.25/M]
  价格换算，美元：0.000955，人民币：0.006876
  ```

**输入参数**:
- `api_key` (STRING): ZenMux 平台的 API Key（在 zenmux.ai 控制台获取）
- `model` (选择): 模型（带价格标注），默认 `openai/gpt-5.4-nano`
- `system_prompt` (STRING): 系统提示词
- `user_prompt` (STRING): 用户提示词
- `seed` (INT): 随机种子
- `temperature` (FLOAT, 可选): 采样温度，默认 0.7，范围 0.0 ~ 2.0
- `top_p` (FLOAT, 可选): 核采样阈值，默认 1.0
- `max_tokens` (INT, 可选): 最大输出 token 数，默认 1024
- `image_1` ~ `image_6` (IMAGE, 可选): 多模态图像输入（所选模型需支持 image 输入）
- `detail` (选择, 可选): 图像分析细节等级，auto/low/high
- `image_max_size` (INT, 可选): 发送前图像最长边缩放上限，默认 1024
- `base_url` (STRING, 可选): API 地址，默认 `zenmux.ai/api/v1`（无需写 `https://`，节点会自动补全）
- `proxy_url` (STRING, 可选): HTTP/HTTPS 代理地址，如 `127.0.0.1:7890`
- `usd_to_cny` (FLOAT, 可选): 美元兑人民币汇率，默认 7.2，用于 `usage_stats` 的人民币换算，可按当日牌价调整

**输出**:
- `text` (STRING): 模型生成的文本内容
- `model_id` (STRING): 实际调用的模型 id（如 `openai/gpt-5.4-nano`），便于下游记录
- `usage_stats` (STRING): 单次运行的 token 消耗、输出文字数量（按字符计，含标点）与费用统计（四行文本，格式见上）；请求失败时记为 0 消耗，token 数缺失或单价未知的项显示 `?`

**使用场景**:
- 一个 Key 试遍多家厂商的模型，横向对比效果与成本
- 按预算选型：价格就写在下拉列表里，直接挑便宜的
- 调用 Claude / GPT / Gemini / DeepSeek 等做文本生成或图像理解

---

### 19. Markdown转图片 / Markdown To Image

**分类**: `Rui-Node🐶/文本处理📝`

**功能描述**:  
输入 Markdown 文本，输出按阅读器级排版渲染的图片（IMAGE）。视觉规范对标 GitHub / Typora：标题层级字号（2.0/1.5/1.25/1.0/0.875/0.85 倍正文）、H1/H2 底部分隔线、引用左竖条、代码块圆角底色 + 等宽字体 + 语言标签、表格圆角外框 + 表头加粗底色 + 斑马纹 + 列对齐、任务清单勾选框、彩色 Emoji。纯 PIL 实现，无额外依赖。

**支持的 Markdown 语法**:  
标题 `#`~`######`、段落（单换行即硬换行）、**粗体**、*斜体*、`行内代码`、~~删除线~~、[链接]()、有序/无序/嵌套列表、任务清单 `- [x]`、引用 `>`（可嵌套）、围栏代码块、表格（`:---:` 对齐语法）、分割线 `---`；表格与正文中的 Emoji 以系统彩色字体渲染。

**输入参数**:
- `markdown` (STRING): Markdown 文本
- `size_preset` (选择): 常用尺寸快选（1080×1440 / 1080×1920 / 1080×1080 / 1920×1080 / A4 等），选 `custom` 时使用下方宽高
- `width` / `height` (INT): 精确尺寸（64~8192px，`custom` 时生效）
- `font` (选择): 字体，列表来自 `Ruinode/font` 目录（ttf/otf/ttc 均可，放入后刷新页面即出现在下拉；同族粗体文件如 `msyhbd` 会自动配对用于渲染粗体，无粗体文件时描边模拟）
- `theme` (选择): 浅色 / 深色 / 米色三套阅读器配色
- `body_size`、`h1_size` ~ `h6_size` (STRING, 可选): 各级字号，`auto`（默认）按输出尺寸二分搜索「恰好优雅填满画布」的字号，各级也可分别填数字精确指定
- `letter_spacing` (STRING, 可选): 字间距像素，默认 `auto`
- `line_spacing` (STRING, 可选): 行高倍数（如 `1.8`），默认 `auto`（正文 1.65）
- `max_chars_per_line` (STRING, 可选): 单行文字字数上限，达到即换行；默认 `auto`（按像素宽自然换行）

**输出**:
- `image` (IMAGE): 渲染结果，固定为所选宽高；内容超高时按现有字号裁剪并在控制台提示

**使用场景**:
- 将 LLM 输出的 Markdown（如 ZenMux 节点的 text）直接转成可分享的长图
- 生成小红书 / 公众号风格的图文卡片、A4 打印稿
- 工作流内把结构化报告（含表格、代码）落成图像资产

**⚠️ 重要：不要用 WAS 的「Text Multiline」节点喂 Markdown**  
WAS Node Suite 的「Text Multiline」会把 `#` 开头的行**当注释删除**，标题行会凭空消失，还会做动态提示词替换。请改用本套件的 [多行文本框(原样输出)](#20-多行文本框原样输出--text-box-raw)，或直接在本节点的 `markdown` 输入框里粘贴文本。

---

### 20. 多行文本框(原样输出) / Text Box (Raw)

**分类**: `Rui-Node🐶/文本处理📝`

**功能描述**:  
把输入的多行文本**一字不动**输出为 STRING：不删注释行、不做动态提示词/通配符/token 替换。专门用来安全承载 Markdown、代码等格式敏感文本（WAS 的「Text Multiline」会把 `#` 开头的行当注释吃掉，喂 Markdown 时标题会消失）。

**输入参数**:
- `text` (STRING): 多行文本

**输出**:
- `text` (STRING): 与输入完全一致的文本

**使用场景**:
- 为 [Markdown转图片](#19-markdown转图片--markdown-to-image) 提供含 `#` 标题的原样文本
- 存放任何不希望被上游文本节点"加工"的内容

---

## 🐕 关于 Rui-Node🐶

Rui-Node🐶 致力于为 ComfyUI 用户提供实用、高效的节点工具集。🐶 是我们的项目标志，代表着忠诚、友好和可靠。

## 📄 许可证

本项目遵循开源协议，欢迎使用和贡献。

---

**Happy Creating with Rui-Node🐶!** 🎨✨
