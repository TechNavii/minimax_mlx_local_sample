# MiniMax M2 Local Server (MLX)

A minimal local HTTP server that runs a Hugging Face chat or text generation model with Apple’s MLX runtime (optimized for Apple Silicon). It exposes simple JSON and OpenAI-compatible endpoints for non-streaming and SSE streaming responses.

## Requirements
- Python 3.10+
- macOS on Apple Silicon (MLX targets M‑series GPUs/CPUs)
- A supported HF model repo (e.g., `mlx-community/MiniMax-M2-4bit`)

## Quickstart

1) Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies
```
pip install -U pip
pip install -r requirements.txt
```

3) Run the server
```
export MODEL_REPO="mlx-community/MiniMax-M2-4bit"
export SERVER=1
# Optional:
# export HOST=0.0.0.0
# export PORT=8000
python app.py
```

4) Try a request (non-streaming)
```
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 128,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": false
      }'
```

For raw text generation:
```
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Write a haiku about autumn", "max_tokens": 128}'
```

To enable server-sent events streaming, set `"stream": true` in the request.

## Endpoints
- `POST /generate` or `/v1/generate`
  - Body: `{ prompt?: string, messages?: Message[], max_tokens?: number, temperature?: number, top_p?: number, stream?: boolean }`
  - Returns generated text; when `stream=true`, sends SSE chunks `{"text": "..."}` and terminates with `[DONE]`.

- `POST /v1/completions`
  - Returns OpenAI-style text completion object; supports `stream` for SSE chunked responses.

- `POST /v1/chat/completions`
  - Accepts OpenAI-style `messages` array and returns a chat completion; supports `stream` for SSE.

- `POST /shutdown` or `/v1/shutdown`
  - Gracefully stops the server and frees resources.

Notes:
- Use `LOG=1` to print brief request debug info to stderr.
- If your tokenizer defines a chat template, you can provide `messages` instead of `prompt`.

## Environment Variables
- `MODEL_REPO` (required): Hugging Face repo id (e.g., `mlx-community/MiniMax-M2-4bit`).
- `MODEL_ID` (optional): Overrides model id exposed by OpenAI-compatible endpoints.
- `SERVER` (optional): When `1|true|yes` or `--serve` CLI flag present, runs HTTP server.
- `HOST` (optional): Bind address (default `127.0.0.1`).
- `PORT` (optional): Port (default `8000`).
- `LOG` (optional): When `1|true|yes`, prints short request previews.

## One-off demo (non-server)
If you don’t set `SERVER` or `--serve`, running `python app.py` will perform a single generation and exit.

## Model support
The app performs a quick preflight on the HF model config to avoid downloading unsupported architectures. If your model isn’t supported by `mlx_lm`, pick a different repo or load it via Transformers with `trust_remote_code=True` on MPS/CPU.

## Development
- Format/editor settings are standardized with `.editorconfig`.
- Python/macos-specific ignores live in `.gitignore`.
- For issues and feature requests, use the provided templates under `.github/`.

## License
This project is licensed under the MIT License (see `LICENSE`). You may change the license to suit your needs.

