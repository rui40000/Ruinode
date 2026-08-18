# -*- coding: utf-8 -*-
"""
加载透明视频节点（Alpha）—— Ruinode
====================================
从带透明通道的视频中解出 RGBA 序列帧，alpha 不丢失。

为什么需要这个节点：
带 alpha 的 WebM（VP8/VP9）并不把透明度存在主视频流里——主流仍是
yuv420p，alpha 被单独压成第二路，藏在 Matroska 的 BlockAdditional
边带中，容器上只留一条 alpha_mode=1 的元数据作记号。
ffmpeg 内置的 vp9 / vp8 解码器不读这条边带，只有 libvpx 系列
（libvpx-vp9 / libvpx）才会。VideoHelperSuite 等常见加载节点走的是
默认解码器，因此拿到的每一帧 alpha 恒为 255，看起来「透明通道丢了」。

本节点显式指定 libvpx 解码器，并以 rgba 原始像素流读回，从而完整
保留透明度（含半透明边缘）。对 MOV/qtrle、ProRes 4444 等本身就带
alpha 通道的格式同样适用。

输出的 rgba_image 是 4 通道 IMAGE，可直接接 ComfyUI 原生「保存图像」
节点存成带透明通道的 PNG 序列帧。
"""

import os
import re
import shutil
import subprocess

import numpy as np
import torch

try:
    import folder_paths
except Exception:      # 脱离 ComfyUI 单测时
    folder_paths = None


VIDEO_EXTS = (".webm", ".mkv", ".mov", ".mp4", ".avi", ".gif", ".apng", ".m4v")

# 这些编码把 alpha 存在边带里，必须换 libvpx 系解码器才读得到
LIBVPX_DECODER = {"vp9": "libvpx-vp9", "vp8": "libvpx"}

# pix_fmt 前缀带 alpha 的情形（ProRes 4444、qtrle、rgba 原始流等）
ALPHA_PIX_PREFIX = ("yuva", "rgba", "bgra", "argb", "abgr", "ya8", "ya16",
                    "gbrap", "rgba64", "bgra64")


def _find_ffmpeg():
    """按可靠性依次寻找可用的 ffmpeg。"""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # VideoHelperSuite 装过的话通常也带一份
    try:
        from videohelpersuite import utils as vhs_utils
        exe = getattr(vhs_utils, "ffmpeg_path", None)
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    raise RuntimeError(
        "找不到 ffmpeg。请安装 imageio-ffmpeg（pip install imageio-ffmpeg），"
        "或把 ffmpeg 加入系统 PATH。")


def _probe(path):
    """探测视频的编码、尺寸、帧率与是否带 alpha。

    优先用 PyAV（元数据最准），不可用时回退到解析 ffmpeg -i 的输出。
    """
    info = {"codec": "", "width": 0, "height": 0, "fps": 0.0,
            "has_alpha": False, "pix_fmt": ""}
    try:
        import av
        with av.open(path) as c:
            st = c.streams.video[0]
            cc = st.codec_context
            info["codec"] = (cc.name or "").lower()
            info["width"] = int(cc.width or 0)
            info["height"] = int(cc.height or 0)
            info["pix_fmt"] = (cc.pix_fmt or "").lower()
            try:
                info["fps"] = float(st.average_rate) if st.average_rate else 0.0
            except Exception:
                info["fps"] = 0.0
            meta = {}
            meta.update(st.metadata or {})
            meta.update(c.metadata or {})
            alpha_mode = str(meta.get("alpha_mode", "")).strip()
            info["has_alpha"] = (alpha_mode == "1") or any(
                info["pix_fmt"].startswith(p) for p in ALPHA_PIX_PREFIX)
        if info["width"] and info["height"]:
            return info
    except Exception:
        pass

    # 回退：ffmpeg -i 把流信息打在 stderr 上
    try:
        p = subprocess.run([_find_ffmpeg(), "-i", path],
                           capture_output=True, text=True, errors="ignore")
        err = p.stderr
        m = re.search(r"Stream #\d+:\d+.*?: Video: (\w+).*?, (\w+).*?, (\d+)x(\d+)",
                      err, re.S)
        if m:
            info["codec"] = m.group(1).lower()
            info["pix_fmt"] = m.group(2).lower()
            info["width"], info["height"] = int(m.group(3)), int(m.group(4))
        m = re.search(r"(\d+(?:\.\d+)?) fps", err)
        if m:
            info["fps"] = float(m.group(1))
        info["has_alpha"] = (
            re.search(r"alpha_mode\s*:\s*1", err) is not None
            or any(info["pix_fmt"].startswith(p) for p in ALPHA_PIX_PREFIX))
    except Exception:
        pass
    return info


def _read_exact(pipe, n):
    """管道一次 read 未必给满一帧，循环补齐。"""
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def _list_videos():
    if folder_paths is None:
        return []
    try:
        d = folder_paths.get_input_directory()
        return sorted(f for f in os.listdir(d)
                      if f.lower().endswith(VIDEO_EXTS))
    except Exception:
        return []


class RuiLoadVideoAlpha:
    """加载带透明通道的视频，输出 RGBA 序列帧。"""

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_videos()
        return {
            "required": {
                "video": (files if files else ["（input 目录没有视频）"], {
                    "video_upload": True,
                    "tooltip": "从 ComfyUI 的 input 目录里选择视频文件。\n"
                               "若视频不在该目录，可改用下方「视频路径」直接填绝对路径。"
                }),
                "强制帧率": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "按指定帧率重采样。填 0 表示保持视频原始帧率。\n"
                               "例如原视频 8fps、这里填 24，会补成 24fps（重复帧）；\n"
                               "填 4 则抽帧减半。做游戏序列帧时通常保持 0 更安全。"
                }),
                "帧数上限": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "最多读取多少帧，填 0 表示读完整段。\n"
                               "达到上限会立即中止解码，用来快速试跑长视频。"
                }),
                "跳过前N帧": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "丢弃视频开头的若干帧。\n"
                               "用于跳过片头黑场或起手多余的静止帧。"
                }),
                "间隔": ("INT", {
                    "default": 1, "min": 1, "max": 100, "step": 1,
                    "tooltip": "每隔几帧取一帧。1=每帧都要，2=隔帧抽取（帧数减半）。\n"
                               "在「跳过前N帧」之后生效。"
                }),
                "自定义宽度": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "输出宽度，填 0 保持原始尺寸。\n"
                               "只填宽或只填高时，另一边按原比例自动换算。"
                }),
                "自定义高度": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 8,
                    "tooltip": "输出高度，填 0 保持原始尺寸。\n"
                               "像素素材缩放会让边缘被插值糊掉，做像素游戏时建议保持 0。"
                }),
                "解码器": (["自动", "强制 libvpx（保 alpha）", "默认解码器"], {
                    "default": "自动",
                    "tooltip": "自动：探测到 alpha_mode=1 的 VP8/VP9 时自动换用 libvpx，\n"
                               "  否则用默认解码器（更快）。一般保持「自动」即可。\n\n"
                               "强制 libvpx：无论探测结果如何都用 libvpx 解码。\n"
                               "  当自动模式仍然丢 alpha 时改用这一档。\n\n"
                               "默认解码器：用 ffmpeg 内置解码器，速度快但\n"
                               "  读不到 WebM 边带里的 alpha（这正是常见加载\n"
                               "  节点丢透明通道的原因）。"
                }),
            },
            "optional": {
                "视频路径": ("STRING", {
                    "default": "",
                    "tooltip": "视频文件的绝对路径。填写后优先于上方的下拉选择。\n"
                               "留空则使用下拉框选中的文件。"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("rgba_image", "alpha", "rgb_image", "帧数", "帧率")
    OUTPUT_TOOLTIPS = (
        "4 通道 RGBA 序列帧。直接接 ComfyUI 原生「保存图像」即可存成\n"
        "带透明通道的 PNG 序列帧（原生保存节点不会丢 alpha）。",
        "透明通道单独输出为遮罩，可接遮罩预览或参与后续合成。",
        "3 通道 RGB 序列帧，供只接受 3 通道输入的下游节点使用。",
        "实际读取到的帧数。",
        "实际输出的帧率（未设强制帧率时即视频原始帧率）。",
    )
    FUNCTION = "load"
    CATEGORY = "Rui-Node🐶/视频🎬"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        path = (kwargs.get("视频路径") or "").strip()
        if not path and folder_paths is not None:
            try:
                path = folder_paths.get_annotated_filepath(kwargs.get("video", ""))
            except Exception:
                path = ""
        if path and os.path.isfile(path):
            return os.path.getmtime(path)
        return float("nan")

    def load(self, **kwargs):
        video = kwargs.get("video", "")
        custom_path = (kwargs.get("视频路径") or "").strip().strip('"')
        force_rate = float(kwargs.get("强制帧率", 0.0))
        cap = int(kwargs.get("帧数上限", 0))
        skip = int(kwargs.get("跳过前N帧", 0))
        every = max(1, int(kwargs.get("间隔", 1)))
        cw = int(kwargs.get("自定义宽度", 0))
        ch = int(kwargs.get("自定义高度", 0))
        dec_mode = kwargs.get("解码器", "自动")

        # ── 定位文件 ──
        if custom_path:
            path = custom_path
        else:
            if folder_paths is None:
                raise RuntimeError("未在 ComfyUI 环境中运行，请填写「视频路径」。")
            path = folder_paths.get_annotated_filepath(video)
        if not path or not os.path.isfile(path):
            raise RuntimeError("视频文件不存在：%s" % path)

        info = _probe(path)
        src_w, src_h = info["width"], info["height"]
        if not src_w or not src_h:
            raise RuntimeError("无法解析视频尺寸：%s" % path)

        # ── 选择解码器 ──
        # 这是本节点的核心：WebM 的 alpha 藏在 BlockAdditional 边带里，
        # 只有 libvpx 系解码器会去读它，内置 vp9/vp8 解码器直接无视。
        decoder = None
        if dec_mode == "强制 libvpx（保 alpha）":
            decoder = LIBVPX_DECODER.get(info["codec"])
        elif dec_mode == "自动":
            if info["has_alpha"]:
                decoder = LIBVPX_DECODER.get(info["codec"])
        # 「默认解码器」保持 decoder = None

        # ── 计算输出尺寸 ──
        if cw > 0 and ch > 0:
            out_w, out_h = cw, ch
        elif cw > 0:
            out_w = cw
            out_h = max(1, int(round(src_h * cw / src_w)))
        elif ch > 0:
            out_h = ch
            out_w = max(1, int(round(src_w * ch / src_h)))
        else:
            out_w, out_h = src_w, src_h

        # ── 组装解码命令 ──
        ff = _find_ffmpeg()
        cmd = [ff, "-loglevel", "error", "-nostdin"]
        if decoder:
            cmd += ["-c:v", decoder]          # 必须在 -i 之前才生效
        cmd += ["-i", path]
        if force_rate > 0:
            cmd += ["-r", str(force_rate)]
        if (out_w, out_h) != (src_w, src_h):
            cmd += ["-vf", "scale=%d:%d:flags=lanczos" % (out_w, out_h)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgba", "-"]

        frame_bytes = out_w * out_h * 4
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        raw_frames = []
        idx = kept = 0
        err = ""
        try:
            while True:
                buf = _read_exact(proc.stdout, frame_bytes)
                if len(buf) < frame_bytes:
                    break
                if idx >= skip and (idx - skip) % every == 0:
                    raw_frames.append(
                        np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 4))
                    kept += 1
                    if cap > 0 and kept >= cap:
                        break
                idx += 1
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.terminate()
            try:
                err = proc.stderr.read().decode("utf-8", "ignore")
                proc.stderr.close()
            except Exception:
                pass
            proc.wait()

        if not raw_frames:
            raise RuntimeError(
                "未能从视频解出任何帧。\n文件：%s\n编码：%s  解码器：%s\n"
                "ffmpeg 输出：%s"
                % (path, info["codec"], decoder or "默认",
                   err.strip()[:400] or "（无）"))

        # ── 转张量：预分配后逐帧写入并及时释放，避免同时堆两份大数组 ──
        n = len(raw_frames)
        rgba = torch.empty((n, out_h, out_w, 4), dtype=torch.float32)
        for i in range(n):
            rgba[i] = torch.from_numpy(raw_frames[i].copy()).float().div_(255.0)
            raw_frames[i] = None

        alpha = rgba[..., 3].contiguous()
        rgb = rgba[..., :3].contiguous()
        fps = force_rate if force_rate > 0 else (info["fps"] or 0.0)
        if every > 1 and fps > 0:
            fps = fps / every

        a_min, a_max = float(alpha.min()), float(alpha.max())
        if a_min >= 1.0:
            print("[Ruinode 加载透明视频] 注意：解出的 alpha 全为不透明。"
                  "编码=%s 解码器=%s 探测到alpha=%s。"
                  "若该视频确实带透明通道，请把「解码器」切到"
                  "「强制 libvpx（保 alpha）」再试。"
                  % (info["codec"], decoder or "默认", info["has_alpha"]))
        else:
            print("[Ruinode 加载透明视频] %d 帧 %dx%d 解码器=%s "
                  "alpha范围=[%.3f,%.3f] 全透明占比=%.3f"
                  % (n, out_w, out_h, decoder or "默认", a_min, a_max,
                     float((alpha < 0.004).float().mean())))

        return (rgba, alpha, rgb, n, float(fps))


NODE_CLASS_MAPPINGS = {
    "RuiLoadVideoAlpha": RuiLoadVideoAlpha,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RuiLoadVideoAlpha": "加载透明视频 / Load Video (Alpha)",
}
