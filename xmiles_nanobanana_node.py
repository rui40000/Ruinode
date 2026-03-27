import torch
import numpy as np
import requests
import json
import uuid
import io
import base64
import time
from PIL import Image

class XmilesNanobananaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "resolution": (["1K", "2K", "4K"], {"default": "4K"}),
                "aspect_ratio": (["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"], {"default": "9:16"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "proxy_url": ("STRING", {"default": "", "multiline": False}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "log")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "generate"
    CATEGORY = "Rui-Node🐶/AI模型🤖"

    def _make_client_id(self):
        return f"{uuid.uuid4()}-test"

    def _tensor_to_png_base64(self, tensor):
        arr = tensor.cpu().numpy()
        arr = np.clip(arr, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8), 'RGB')
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _pil_to_tensor(self, pil_img):
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        np_img = np.array(pil_img).astype(np.float32) / 255.0
        t = torch.from_numpy(np_img).unsqueeze(0)
        return t

    def _download_image_tensor(self, url, proxies=None):
        r = requests.get(url, proxies=proxies, timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        return self._pil_to_tensor(img)

    def generate(self, text, resolution, aspect_ratio, images=None, proxy_url="", verbose=True):
        t0 = time.perf_counter()
        logs = []
        if verbose: 
            print("Xmiles-nanobanana:start", {"ts": t0, "resolution": resolution, "aspect_ratio": aspect_ratio}, flush=True)
        logs.append(f"start_ts={t0}")
        client_id = self._make_client_id()
        if verbose: 
            print("Xmiles-nanobanana:client_id", client_id, flush=True)
        logs.append(f"client_id={client_id}")
        proxies = None
        if proxy_url and proxy_url.strip():
            proxies = {"http": proxy_url, "https": proxy_url}
            if verbose: 
                print("Xmiles-nanobanana:proxies", proxies, flush=True)
            logs.append(f"proxies={proxy_url}")
        else:
            if verbose: 
                print("Xmiles-nanobanana:proxies=none", flush=True)
            logs.append("proxies=none")

        parts = []
        if text and text.strip():
            parts.append({"text": text})
            if verbose: 
                print("Xmiles-nanobanana:text_len", len(text), flush=True)
            logs.append(f"text_len={len(text)}")

        image_parts = []
        if images is not None:
            if isinstance(images, list):
                tensors = [img[0] for img in images]
            else:
                tensors = [images[0]]
            if verbose: 
                print("Xmiles-nanobanana:image_count", len(tensors), flush=True)
            logs.append(f"image_count={len(tensors)}")
            for t in tensors:
                b64 = self._tensor_to_png_base64(t)
                image_parts.append({"inlineData": {"data": "data:image/png;base64," + b64, "mimeType": "image/png"}})

        for p in image_parts:
            parts.append(p)

        t1 = time.perf_counter()
        if verbose: 
            print("Xmiles-nanobanana:parts_ready_ms", int((t1 - t0) * 1000), flush=True)
        logs.append(f"parts_ready_ms={(t1-t0)*1000:.2f}")

        body_obj = {
            "contents": [
                {
                    "parts": parts,
                    "role": "user"
                }
            ],
            "filePath": {
                "inputs": {
                    "filePath": ""
                }
            },
            "generationConfig": {
                "candidateCount": 1,
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution
                },
                "responseModalities": ["TEXT", "IMAGE"],
                "temperature": 1.0,
                "topP": 0.95
            },
            "model": "gemini-3.1-flash-image-preview"
        }

        t2 = time.perf_counter()
        if verbose: 
            print("Xmiles-nanobanana:body_size", len(json.dumps(body_obj, ensure_ascii=False)), flush=True)
        logs.append(f"body_size={len(json.dumps(body_obj, ensure_ascii=False))}")

        payload = {
            "taskType": "ZENMUX",
            "clientId": client_id,
            "clientType": "image",
            "callBackService": "remoteApi",
            "extraData": {
                "faceDetailer": 0,
                "filePath": "",
                "loraNum": 0,
                "memberType": "PLUS",
                "moduleName": "全能编辑 V2",
                "resolution": "",
                "taskType": "ZENMUX",
                "uniqueId": str(uuid.uuid4().int)[:19],
                "workflowName": ""
            },
            "imgIdList": [],
            "memberType": "PLUS",
            "body": json.dumps(body_obj, ensure_ascii=False)
        }

        url = "https://test.holopix.cn/ai-holopix-queue/api/prompt"
        try:
            t3 = time.perf_counter()
            if verbose: 
                print("Xmiles-nanobanana:post_begin", {"ts": t3, "url": url}, flush=True)
            logs.append(f"post_begin_ts={t3}")
            resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, proxies=proxies, timeout=60)
            t4 = time.perf_counter()
            resp.raise_for_status()
            data = resp.json()
            if verbose:
                print("Xmiles-nanobanana:post_done", {"status_code": resp.status_code, "elapsed_ms": int((t4 - t3) * 1000)}, flush=True)
            logs.append(f"post_elapsed_ms={(t4-t3)*1000:.2f}")
            status = data.get("status")
            gen_status = data.get("generateStatus")
            logs = {"status": status, "generateStatus": gen_status, "clientId": data.get("clientId"), "timestamp": data.get("timestamp")}
            tensors = []
            if status == 0 and gen_status == 1:
                items = data.get("data") or []
                if verbose:
                    print("Xmiles-nanobanana:result_items", len(items), flush=True)
                for item in items:
                    url_item = item.get("url")
                    if url_item:
                        d0 = time.perf_counter()
                        if verbose:
                            print("Xmiles-nanobanana:download_begin", url_item, flush=True)
                        t = self._download_image_tensor(url_item, proxies=proxies)
                        tensors.append(t)
                        d1 = time.perf_counter()
                        if verbose:
                            print("Xmiles-nanobanana:download_done_ms", int((d1 - d0) * 1000), flush=True)
                return (tensors if tensors else [], json.dumps(logs, ensure_ascii=False))
            else:
                if verbose:
                    print("Xmiles-nanobanana:task_failed", {"status": status, "generateStatus": gen_status}, flush=True)
                return ([], json.dumps(data, ensure_ascii=False))
        except Exception as e:
            if verbose:
                print("Xmiles-nanobanana:error", str(e), flush=True)
            return ([], str(e))

class XmilesNanobananaResultParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True, "default": ""}),
                "proxy_url": ("STRING", {"default": "", "multiline": False}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "log")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "parse"
    CATEGORY = "Rui-Node🐶/AI模型🤖"

    def _download_image_tensor(self, url, proxies=None):
        r = requests.get(url, proxies=proxies, timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        np_img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(np_img).unsqueeze(0)

    def parse(self, json_text, proxy_url="", verbose=True):
        p0 = time.perf_counter()
        if verbose:
            print("Xmiles-nanobanana:parse_begin", {"ts": p0}, flush=True)
        proxies = None
        if proxy_url and proxy_url.strip():
            proxies = {"http": proxy_url, "https": proxy_url}
            if verbose:
                print("Xmiles-nanobanana:parse_proxies", proxies, flush=True)
        try:
            obj = json.loads(json_text)
            status = obj.get("status")
            gen_status = obj.get("generateStatus")
            if verbose:
                print("Xmiles-nanobanana:parse_status", {"status": status, "generateStatus": gen_status}, flush=True)
            tensors = []
            if status == 0 and gen_status == 1:
                items = obj.get("data") or []
                if verbose:
                    print("Xmiles-nanobanana:parse_items", len(items), flush=True)
                for item in items:
                    url_item = item.get("url")
                    if url_item:
                        z0 = time.perf_counter()
                        if verbose:
                            print("Xmiles-nanobanana:parse_download_begin", url_item, flush=True)
                        tensors.append(self._download_image_tensor(url_item, proxies=proxies))
                        z1 = time.perf_counter()
                        if verbose:
                            print("Xmiles-nanobanana:parse_download_ms", int((z1 - z0) * 1000), flush=True)
            return (tensors, json.dumps({"status": status, "generateStatus": gen_status}, ensure_ascii=False))
        except Exception as e:
            if verbose:
                print("Xmiles-nanobanana:parse_error", str(e), flush=True)
            return ([], str(e))

NODE_CLASS_MAPPINGS = {
    "XmilesNanobanana": XmilesNanobananaNode,
    "XmilesNanobananaResultParser": XmilesNanobananaResultParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XmilesNanobanana": "Xmiles-nanobanana",
    "XmilesNanobananaResultParser": "Xmiles-nanobanana 结果解析",
}
