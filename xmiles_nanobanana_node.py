import torch
import numpy as np
import requests
import json
import uuid
import io
import base64
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

    def generate(self, text, resolution, aspect_ratio, images=None, proxy_url=""):
        client_id = self._make_client_id()
        proxies = None
        if proxy_url and proxy_url.strip():
            proxies = {"http": proxy_url, "https": proxy_url}

        parts = []
        if text and text.strip():
            parts.append({"text": text})

        image_parts = []
        if images is not None:
            if isinstance(images, list):
                tensors = [img[0] for img in images]
            else:
                tensors = [images[0]]
            for t in tensors:
                b64 = self._tensor_to_png_base64(t)
                image_parts.append({"inlineData": {"data": "data:image/png;base64," + b64, "mimeType": "image/png"}})

        for p in image_parts:
            parts.append(p)

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
            resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, proxies=proxies, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            gen_status = data.get("generateStatus")
            logs = {"status": status, "generateStatus": gen_status, "clientId": data.get("clientId"), "timestamp": data.get("timestamp")}
            tensors = []
            if status == 0 and gen_status == 1:
                items = data.get("data") or []
                for item in items:
                    url_item = item.get("url")
                    if url_item:
                        t = self._download_image_tensor(url_item, proxies=proxies)
                        tensors.append(t)
                return (tensors if tensors else [], json.dumps(logs, ensure_ascii=False))
            else:
                return ([], json.dumps(data, ensure_ascii=False))
        except Exception as e:
            return ([], str(e))

class XmilesNanobananaResultParser:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_text": ("STRING", {"multiline": True, "default": ""}),
                "proxy_url": ("STRING", {"default": "", "multiline": False}),
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

    def parse(self, json_text, proxy_url=""):
        proxies = None
        if proxy_url and proxy_url.strip():
            proxies = {"http": proxy_url, "https": proxy_url}
        try:
            obj = json.loads(json_text)
            status = obj.get("status")
            gen_status = obj.get("generateStatus")
            tensors = []
            if status == 0 and gen_status == 1:
                items = obj.get("data") or []
                for item in items:
                    url_item = item.get("url")
                    if url_item:
                        tensors.append(self._download_image_tensor(url_item, proxies=proxies))
            return (tensors, json.dumps({"status": status, "generateStatus": gen_status}, ensure_ascii=False))
        except Exception as e:
            return ([], str(e))

NODE_CLASS_MAPPINGS = {
    "XmilesNanobanana": XmilesNanobananaNode,
    "XmilesNanobananaResultParser": XmilesNanobananaResultParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XmilesNanobanana": "Xmiles-nanobanana",
    "XmilesNanobananaResultParser": "Xmiles-nanobanana 结果解析",
}
