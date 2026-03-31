import torch
import numpy as np
import requests
import json
import uuid
import io
import base64
import time
from PIL import Image

try:
    import oss2  # type: ignore
except Exception:
    oss2 = None

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
                "use_oss": ("BOOLEAN", {"default": False}),
                "on_queued": (["passthrough", "empty"], {"default": "passthrough"}),
                "oss_access_key": ("STRING", {"default": "", "multiline": False}),
                "oss_secret_key": ("STRING", {"default": "", "multiline": False}),
                "oss_endpoint": ("STRING", {"default": "", "multiline": False, "placeholder": "oss-cn-shanghai.aliyuncs.com"}),
                "oss_bucket_name": ("STRING", {"default": "", "multiline": False}),
                "oss_url_prefix": ("STRING", {"default": "", "multiline": False, "placeholder": "https://genai.holopix.cn"}),
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

    def _upload_to_oss_and_get_url(self, tensor, ak, sk, endpoint, bucket_name, url_prefix, verbose=False, proxy_url=""):
        if oss2 is None:
            raise RuntimeError("oss2 is not installed. Please `pip install oss2` and try again.")
        arr = tensor.cpu().numpy()
        arr = np.clip(arr, 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8), 'RGB')
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        key = f"comfyui/{uuid.uuid4().hex}.png"
        if verbose:
            print("Xmiles-nanobanana:oss_put_begin", {"key": key, "size": len(data)}, flush=True)
        auth = oss2.Auth(ak, sk)
        endp = endpoint if endpoint.startswith("http") else ("https://" + endpoint) if endpoint else ""
        old_http = None
        old_https = None
        try:
            if proxy_url and proxy_url.strip():
                old_http = os.environ.get("HTTP_PROXY")
                old_https = os.environ.get("HTTPS_PROXY")
                os.environ["HTTP_PROXY"] = proxy_url
                os.environ["HTTPS_PROXY"] = proxy_url
            bucket = oss2.Bucket(auth, endp, bucket_name)
            result = bucket.put_object(key, data)
            if result.status not in (200, 204):
                raise RuntimeError(f"OSS put_object failed, status={result.status}")
        except Exception as e:
            if verbose:
                print("Xmiles-nanobanana:oss_put_error_primary", str(e), flush=True)
            if url_prefix:
                cname_ep = url_prefix if url_prefix.startswith("http") else ("https://" + url_prefix)
                try:
                    bucket = oss2.Bucket(auth, cname_ep, bucket_name, is_cname=True)
                    result = bucket.put_object(key, data)
                    if result.status not in (200, 204):
                        raise RuntimeError(f"CNAME put_object failed, status={result.status}")
                except Exception as e2:
                    if verbose:
                        print("Xmiles-nanobanana:oss_put_error_cname", str(e2), flush=True)
                    raise
            else:
                raise
        finally:
            if proxy_url and proxy_url.strip():
                if old_http is None:
                    os.environ.pop("HTTP_PROXY", None)
                else:
                    os.environ["HTTP_PROXY"] = old_http
                if old_https is None:
                    os.environ.pop("HTTPS_PROXY", None)
                else:
                    os.environ["HTTPS_PROXY"] = old_https
        url = url_prefix.rstrip("/") + "/" + key
        if verbose:
            print("Xmiles-nanobanana:oss_put_done", {"url": url}, flush=True)
        return url

    def _download_image_tensor(self, url, proxies=None):
        r = requests.get(url, proxies=proxies, timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        return self._pil_to_tensor(img)

    def generate(self, text, resolution, aspect_ratio, images=None, proxy_url="", verbose=True, use_oss=False, on_queued="passthrough",
                 oss_access_key="", oss_secret_key="", oss_endpoint="", oss_bucket_name="", oss_url_prefix=""):
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
                if use_oss:
                    if not (oss_access_key and oss_secret_key and oss_endpoint and oss_bucket_name and oss_url_prefix):
                        raise RuntimeError("use_oss=True but OSS credentials/config are missing.")
                    url_uploaded = self._upload_to_oss_and_get_url(
                        t, oss_access_key, oss_secret_key, oss_endpoint, oss_bucket_name, oss_url_prefix, verbose=verbose, proxy_url=proxy_url
                    )
                    image_parts.append({"inlineData": {"data": url_uploaded, "mimeType": "image/png"}})
                else:
                    b64 = self._tensor_to_png_base64(t)
                    image_parts.append({"inlineData": {"data": "data:image/png;base64," + b64, "mimeType": "image/png"}})

        for p in image_parts:
            parts.append(p)

        t1 = time.perf_counter()
        if verbose: 
            print("Xmiles-nanobanana:parts_ready_ms", int((t1 - t0) * 1000), flush=True)
        logs.append(f"parts_ready_ms={(t1-t0)*1000:.2f}")

        unique_id = str(uuid.uuid4().int)[:19]
        if verbose:
            print("Xmiles-nanobanana:unique_id", unique_id, flush=True)
        logs.append(f"unique_id={unique_id}")

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
                "uniqueId": unique_id,
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
            # 补充本地发送的 clientId / uniqueId，便于轮询节点使用
            logs = {
                "status": status,
                "generateStatus": gen_status,
                "clientId": data.get("clientId"),
                "timestamp": data.get("timestamp"),
                "clientId_sent": client_id,
                "uniqueId_sent": unique_id,
                "raw": data
            }
            # 队列受理态识别（部分服务返回 code/success）
            if status is None and gen_status is None and isinstance(data, dict) and data.get("success") is True:
                if verbose:
                    print("Xmiles-nanobanana:queued_accept", {"clientId": client_id, "uniqueId": unique_id}, flush=True)
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
                if on_queued == "passthrough" and images is not None:
                    passthrough = []
                    if isinstance(images, list):
                        passthrough = [img[0] for img in images]
                    else:
                        passthrough = [images[0]]
                    return (passthrough, json.dumps(logs, ensure_ascii=False))
                return ([], json.dumps(logs, ensure_ascii=False))
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

class XmilesNanobananaPoller:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "query_url": ("STRING", {"default": "", "multiline": False}),
                "method": (["GET", "POST"], {"default": "GET"}),
                "client_id": ("STRING", {"default": "", "multiline": False}),
                "unique_id": ("STRING", {"default": "", "multiline": False}),
                "poll_interval_ms": ("INT", {"default": 2000, "min": 200, "max": 60000, "step": 100}),
                "max_wait_ms": ("INT", {"default": 60000, "min": 2000, "max": 600000, "step": 1000}),
            },
            "optional": {
                "payload_template": ("STRING", {"multiline": True, "default": ""}),
                "proxy_url": ("STRING", {"default": "", "multiline": False}),
                "verbose": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "log")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "poll"
    CATEGORY = "Rui-Node🐶/AI模型🤖"

    def _download_image_tensor(self, url, proxies=None):
        r = requests.get(url, proxies=proxies, timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        np_img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(np_img).unsqueeze(0)

    def poll(self, query_url, method, client_id, unique_id, poll_interval_ms, max_wait_ms, payload_template="", proxy_url="", verbose=True):
        t_start = time.perf_counter()
        proxies = None
        if proxy_url and proxy_url.strip():
            proxies = {"http": proxy_url, "https": proxy_url}
        last_obj = None
        while (time.perf_counter() - t_start) * 1000 < max_wait_ms:
            try:
                body = None
                headers = {"Content-Type": "application/json"}
                if payload_template and payload_template.strip():
                    body_str = payload_template.replace("{clientId}", client_id).replace("{uniqueId}", unique_id)
                    try:
                        body = json.loads(body_str)
                    except:
                        body = body_str
                if verbose:
                    print("Xmiles-nanobanana:poll_tick", {"url": query_url}, flush=True)
                if method == "POST":
                    r = requests.post(query_url, headers=headers, json=body if isinstance(body, dict) else None, data=None if isinstance(body, dict) else body, proxies=proxies, timeout=30)
                else:
                    params = {"clientId": client_id, "uniqueId": unique_id}
                    r = requests.get(query_url, headers=headers, params=params, proxies=proxies, timeout=30)
                r.raise_for_status()
                obj = r.json()
                last_obj = obj
                status = obj.get("status")
                gen_status = obj.get("generateStatus")
                if verbose:
                    print("Xmiles-nanobanana:poll_status", {"status": status, "generateStatus": gen_status}, flush=True)
                if status == 0 and gen_status == 1:
                    tensors = []
                    items = obj.get("data") or []
                    for item in items:
                        url_item = item.get("url")
                        if url_item:
                            tensors.append(self._download_image_tensor(url_item, proxies=proxies))
                    return (tensors, json.dumps(obj, ensure_ascii=False))
            except Exception as e:
                if verbose:
                    print("Xmiles-nanobanana:poll_error", str(e), flush=True)
            time.sleep(poll_interval_ms / 1000.0)
        return ([], json.dumps(last_obj if last_obj is not None else {"error": "timeout"}, ensure_ascii=False))

NODE_CLASS_MAPPINGS = {
    "XmilesNanobanana": XmilesNanobananaNode,
    "XmilesNanobananaResultParser": XmilesNanobananaResultParser,
    "XmilesNanobananaPoller": XmilesNanobananaPoller,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XmilesNanobanana": "Xmiles-nanobanana",
    "XmilesNanobananaResultParser": "Xmiles-nanobanana 结果解析",
    "XmilesNanobananaPoller": "Xmiles-nanobanana 轮询查询",
}
