# font 字体目录

「Markdown转图片」节点的字体候选列表来自本目录，把 `.ttf` / `.otf` / `.ttc` 字体文件放进来、
刷新 ComfyUI 页面即可出现在节点的 `font` 下拉里。

- **字体文件不随仓库分发**（微软雅黑等系统字体受版权保护，且体积大），请自行放入。
  Windows 用户可直接从 `C:\Windows\Fonts` 复制常用字体到本目录，例如：
  `msyh.ttc`（微软雅黑）、`msyhbd.ttc`（雅黑粗体）、`simhei.ttf`（黑体）、
  `simsun.ttc`（宋体）、`simkai.ttf`（楷体）、`Deng.ttf`（等线）等。
- **粗体自动配对**：同目录下存在同名 + `bd`/`b` 后缀的字体（如 `msyh` + `msyhbd`）时，
  渲染粗体会使用真实字重；否则用描边模拟。
- **目录为空时**：节点自动回退使用 Windows 系统字体（微软雅黑/黑体/宋体/Arial）。
- Emoji 彩色渲染使用系统 `seguiemj.ttf`，代码块等宽字体使用系统 `consola.ttf`，无需放入本目录。
