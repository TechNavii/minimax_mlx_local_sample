from mlx_lm import load, generate, stream_generate
from mlx_lm.generate import make_sampler
import os
import sys
import json
import time
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, List, Dict, Any


def preflight_model_support(hf_repo: str) -> Optional[str]:
    """Return model_type from config if available; exit early if unsupported by mlx_lm.

    This avoids downloading large weights for unsupported architectures (e.g., "minimax").
    """
    try:
        from huggingface_hub import hf_hub_download
        import json
        import pkgutil
        import mlx_lm.models as mlx_models

        cfg_path = hf_hub_download(hf_repo, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type")

        supported = {name for _, name, _ in pkgutil.iter_modules(mlx_models.__path__)}
        if model_type and model_type not in supported:
            print(
                f"ERROR: Model type '{model_type}' is not supported by mlx_lm.\n"
                f"- Repo: {hf_repo}\n"
                f"- Supported types include examples like: llama, qwen2, mistral3, phi3, etc.\n"
                "Tip: Use a model with a supported model_type, or load this repo via Transformers\n"
                "with trust_remote_code=True on MPS/CPU."
            )
            sys.exit(1)
        return model_type
    except Exception:
        # If preflight fails (offline or missing hub), continue and let mlx_lm handle it.
        return None


# Global state for server mode
MODEL = None
TOKENIZER = None
GEN_LOCK = threading.Lock()
SHUTDOWN_EVENT = threading.Event()
MODEL_ID_STR = None


def load_model_and_tokenizer(hf_repo: str):
    global MODEL, TOKENIZER
    preflight_model_support(hf_repo)
    MODEL, TOKENIZER = load(hf_repo)


def to_prompt(messages: Optional[List[Dict[str, str]]], fallback_prompt: str = "hello") -> str:
    if messages and getattr(TOKENIZER, "chat_template", None):
        return TOKENIZER.apply_chat_template(messages, add_generation_prompt=True)
    return fallback_prompt


def run_generation(prompt: str, max_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95) -> str:
    with GEN_LOCK:
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
        text = generate(
            MODEL,
            TOKENIZER,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
    return text


def release_resources():
    # Ensure MLX finishes queued work and free caches
    try:
        import gc
        import mlx.core as mx

        mx.synchronize()
        mx.clear_cache()
        gc.collect()
    except Exception:
        pass
    finally:
        global MODEL, TOKENIZER
        MODEL = None
        TOKENIZER = None


class LLMHandler(BaseHTTPRequestHandler):
    server_version = "MiniMaxM2-Server/1.0"

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return {}

    def _send(self, code: int, payload: Dict[str, Any]):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Basic CORS for local tools
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        # Optional debug logging
        if os.environ.get("LOG", "0") in ("1", "true", "yes"):
            try:
                ln = int(self.headers.get("Content-Length", "0"))
                preview = self.rfile.peek(min(ln, 512)) if hasattr(self.rfile, 'peek') else b''
                sys.stderr.write(f"POST {self.path} len={ln} body[0:512]={preview!r}\n")
            except Exception:
                sys.stderr.write(f"POST {self.path} len=?\n")

        if self.path in ("/generate", "/v1/generate"):
            req = self._read_json()
            prompt = req.get("prompt")
            messages = req.get("messages")
            if not prompt and messages:
                prompt = to_prompt(messages, fallback_prompt="")
            if not prompt:
                return self._send(400, {"error": "Missing 'prompt' or 'messages'."})

            max_tokens = int(req.get("max_tokens", 256))
            temperature = float(req.get("temperature", 0.7))
            top_p = float(req.get("top_p", 0.95))
            stream = bool(req.get("stream", False))

            if stream:
                return self._sse_generate(prompt, max_tokens, temperature, top_p)
            try:
                output = run_generation(prompt, max_tokens, temperature, top_p)
                return self._send(200, {"text": output})
            except Exception as e:
                return self._send(500, {"error": str(e)})

        if self.path in ("/v1/completions",):
            req = self._read_json()
            prompt = req.get("prompt")
            if not prompt:
                return self._send(400, {"error": "Missing 'prompt'."})
            max_tokens = int(req.get("max_tokens", 256))
            temperature = float(req.get("temperature", 0.7))
            top_p = float(req.get("top_p", 0.95))
            stream = bool(req.get("stream", False))

            if stream:
                return self._sse_completions(prompt, max_tokens, temperature, top_p)

            try:
                output = run_generation(prompt, max_tokens, temperature, top_p)
                now = int(time.time())
                resp = {
                    "id": f"cmpl-{now}",
                    "object": "text_completion",
                    "created": now,
                    "model": MODEL_ID_STR or os.environ.get("MODEL_REPO", "local-model"),
                    "choices": [
                        {
                            "index": 0,
                            "text": output,
                            "finish_reason": "stop",
                        }
                    ],
                }
                return self._send(200, resp)
            except Exception as e:
                return self._send(500, {"error": str(e)})

        if self.path in ("/v1/chat/completions",):
            req = self._read_json()
            messages = req.get("messages", [])
            if not isinstance(messages, list) or not messages:
                return self._send(400, {"error": "'messages' must be a non-empty list."})

            prompt = to_prompt(messages, fallback_prompt="")
            max_tokens = int(req.get("max_tokens", 256))
            temperature = float(req.get("temperature", 0.7))
            top_p = float(req.get("top_p", 0.95))
            stream = bool(req.get("stream", False))

            if stream:
                return self._sse_chat(prompt, max_tokens, temperature, top_p)
            try:
                output = run_generation(prompt, max_tokens, temperature, top_p)
                now = int(time.time())
                resp = {
                    "id": f"chatcmpl-{now}",
                    "object": "chat.completion",
                    "created": now,
                    "model": os.environ.get("MODEL_REPO", "mlx-community/MiniMax-M2-4bit"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": output},
                            "finish_reason": "stop",
                        }
                    ],
                }
                return self._send(200, resp)
            except Exception as e:
                return self._send(500, {"error": str(e)})

        if self.path in ("/shutdown", "/v1/shutdown"):
            # Signal main thread to stop serving
            SHUTDOWN_EVENT.set()
            release_resources()
            return self._send(200, {"status": "shutting_down"})

        return self._send(404, {"error": "Not found"})

    def log_message(self, fmt, *args):
        # Quieter logs
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), time.strftime("%d/%b/%Y %H:%M:%S"), fmt % args))

    # --- Streaming helpers ---
    def _sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # disable proxy buffering
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def _sse_send(self, obj: Dict[str, Any]):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.wfile.write(b"data: ")
        self.wfile.write(data)
        self.wfile.write(b"\n\n")
        try:
            self.wfile.flush()
        except Exception:
            pass

    def _sse_done(self):
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            pass
        # Ensure client sees end-of-stream
        self.close_connection = True

    def _sse_chat(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        model_name = MODEL_ID_STR or os.environ.get("MODEL_REPO", "mlx-community/MiniMax-M2-4bit")
        created = int(time.time())
        self._sse_headers()
        # Initial role-only delta
        first_chunk = {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
        self._sse_send(first_chunk)

        try:
            with GEN_LOCK:
                sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
                for resp in stream_generate(
                    MODEL,
                    TOKENIZER,
                    prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    delta = resp.text
                    if not delta:
                        continue
                    chunk = {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self._sse_send(chunk)
        except BrokenPipeError:
            return
        except Exception as e:
            # Send a terminal error message then close.
            self._sse_send({"error": str(e)})
        finally:
            # Final stop chunk + [DONE]
            stop_chunk = {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            }
            self._sse_send(stop_chunk)
            self._sse_done()

    def _sse_generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        # Simple SSE stream for raw text generation
        self._sse_headers()
        try:
            with GEN_LOCK:
                sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
                for resp in stream_generate(
                    MODEL,
                    TOKENIZER,
                    prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    delta = resp.text
                    if delta:
                        self._sse_send({"text": delta})
        except BrokenPipeError:
            return
        except Exception as e:
            self._sse_send({"error": str(e)})
        finally:
            self._sse_done()

    def _sse_completions(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        model_name = MODEL_ID_STR or os.environ.get("MODEL_REPO", "local-model")
        created = int(time.time())
        self._sse_headers()
        try:
            with GEN_LOCK:
                sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
                for resp in stream_generate(
                    MODEL, TOKENIZER, prompt, max_tokens=max_tokens, sampler=sampler
                ):
                    delta = resp.text
                    if not delta:
                        continue
                    chunk = {
                        "id": f"cmpl-{created}",
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "text": delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                    self._sse_send(chunk)
        except BrokenPipeError:
            return
        except Exception as e:
            self._sse_send({"error": str(e)})
        finally:
            # Final stop marker then [DONE]
            stop_chunk = {
                "id": f"cmpl-{created}",
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {"index": 0, "text": "", "finish_reason": "stop"}
                ],
            }
            self._sse_send(stop_chunk)
            self._sse_done()


def serve(hf_repo: str, host: str = "127.0.0.1", port: int = 8000):
    load_model_and_tokenizer(hf_repo)
    global MODEL_ID_STR
    MODEL_ID_STR = os.environ.get("MODEL_ID", hf_repo)

    httpd = ThreadingHTTPServer((host, port), LLMHandler)
    httpd.timeout = 0.5

    def handle_signals(signum, frame):
        SHUTDOWN_EVENT.set()

    signal.signal(signal.SIGINT, handle_signals)
    signal.signal(signal.SIGTERM, handle_signals)

    try:
        while not SHUTDOWN_EVENT.is_set():
            httpd.handle_request()
    finally:
        release_resources()


def main():
    hf_repo = os.environ.get("MODEL_REPO", "mlx-community/MiniMax-M2-4bit")

    # Mode selection: serve via env SERVER=1 or --serve; else one-off generate
    serve_mode = os.environ.get("SERVER", "0") in ("1", "true", "yes") or "--serve" in sys.argv

    if serve_mode:
        host = os.environ.get("HOST", "127.0.0.1")
        port = int(os.environ.get("PORT", "8000"))
        serve(hf_repo, host=host, port=port)
        return

    # One-off demo generation (previous behavior)
    load_model_and_tokenizer(hf_repo)
    prompt = "hello"
    if getattr(TOKENIZER, "chat_template", None) is not None:
        messages = [{"role": "user", "content": "how are you?"}]
        prompt = TOKENIZER.apply_chat_template(messages, add_generation_prompt=True)
    _ = generate(MODEL, TOKENIZER, prompt=prompt, verbose=True)
    release_resources()


if __name__ == "__main__":
    main()
    def do_OPTIONS(self):
        # CORS preflight support
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()

    def do_GET(self):
        # Health check
        if self.path == "/health":
            return self._send(200, {"status": "ok"})

        # OpenAI-compatible models endpoints
        if self.path == "/v1/models":
            model_id = MODEL_ID_STR or os.environ.get("MODEL_REPO", "local-model")
            return self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": model_id, "object": "model", "owned_by": "local"}
                    ],
                },
            )

        if self.path.startswith("/v1/models/"):
            model_id = self.path.split("/v1/models/", 1)[1]
            return self._send(
                200,
                {"id": model_id, "object": "model", "owned_by": "local"},
            )

        return self._send(404, {"error": "Not found"})
