# Real-ESRGAN Upscaler

A Nuxt 4 web app that upscales images to an exact target resolution with
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), running **entirely in the browser**.
It's a static site deployed to Cloudflare Workers (static assets): no server, no GPU bill, and
images never leave the user's device.

## How it works

1. The official Real-ESRGAN PyTorch weights are converted to ONNX (`scripts/export-onnx.py`) and served
   from `public/models/`. They're split into 20 MB chunks to stay under Cloudflare's 25 MiB per-file limit.
2. A Web Worker ([app/workers/upscaler.worker.ts](app/workers/upscaler.worker.ts)) downloads the model once,
   keeps it in the Cache API, and runs it with [onnxruntime-web](https://onnxruntime.ai/) on **WebGPU**.
   If WebGPU isn't available it falls back to multi-threaded WASM on the CPU. It processes the image in
   overlapping tiles to keep GPU memory bounded.
3. Each AI pass is 4×. A second pass only runs when more than another 1.5× is still needed. Finally
   [pica](https://github.com/nodeca/pica) resizes to the exact resolution with Lanczos3 (stretch, crop or
   pad) and the canvas encodes PNG, JPEG or WebP.

| Model | Download | Notes |
| --- | --- | --- |
| `realesr-general-x4v3` | 5 MB | Default. Fast, good for photos |
| `realesrgan-x4plus` | 67 MB | Highest quality, much slower |
| `realesrgan-x4plus-anime` | 18 MB | Anime / illustrations |
| `realesr-animevideov3` | 2.5 MB | Very fast, anime style |

The onnxruntime WebGPU `.wasm` (26.8 MB) is too big for Cloudflare static assets, so the worker loads the
runtime from jsDelivr, pinned to the installed `onnxruntime-web` version.

Limits: 64 MP output and 16,384 px per side (browser canvas limits), 64 MP for the AI intermediate.

## Development

Requires Node 20+.

```bash
npm install
npm run dev        # http://localhost:3000
npm run preview    # static build served by wrangler dev at http://localhost:8787
```

## Deploy to Cloudflare

```bash
npx wrangler login
npm run deploy     # nuxt generate → wrangler deploy (.output/public as static assets)
```

`public/_headers` sets `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` (needed for
multi-threaded WASM) and long cache lifetimes for models and hashed assets.

## Regenerating the models

The ONNX files in `public/models/` are committed build artefacts. To rebuild them from the official weights:

```bash
python3 -m venv .venv && .venv/bin/pip install torch onnx onnxscript
.venv/bin/python scripts/export-onnx.py
```

If you change a model, bump `MODEL_CACHE` in the worker so browsers drop their cached copy.
