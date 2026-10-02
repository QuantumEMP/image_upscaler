/// <reference lib="webworker" />
import * as ort from 'onnxruntime-web/webgpu'
import type { WorkerRequest, WorkerResponse } from '~/utils/upscaler-types'

// The WebGPU wasm binary is larger than Cloudflare's 25 MiB static-asset limit,
// so load the runtime from jsDelivr (pinned to the installed version, CORS + CORP enabled).
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : 1

const MODEL_CACHE = 'esrgan-models-v1'
const TILE_PAD = 16

let current: { id: string, session: ort.InferenceSession, backend: 'webgpu' | 'wasm' } | undefined

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => (self as DedicatedWorkerGlobalScope).postMessage(msg, transfer)

async function fetchModel(parts: string[], totalBytes: number, onProgress: (f: number) => void) {
  const cache = await caches.open(MODEL_CACHE).catch(() => undefined)
  const out = new Uint8Array(totalBytes)
  let offset = 0
  for (const url of parts) {
    let res = await cache?.match(url)
    if (!res) {
      res = await fetch(url)
      if (!res.ok || !res.body) throw new Error(`Failed to download model (${res.status})`)
      // Stream so we can report progress, then store the chunk in the cache.
      const reader = res.body.getReader()
      const chunks: Uint8Array[] = []
      let size = 0
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        chunks.push(value)
        size += value.length
        onProgress((offset + size) / totalBytes)
      }
      const buf = new Uint8Array(size)
      let o = 0
      for (const c of chunks) { buf.set(c, o); o += c.length }
      await cache?.put(url, new Response(buf)).catch(() => {})
      res = new Response(buf)
    }
    const buf = new Uint8Array(await res.arrayBuffer())
    out.set(buf, offset)
    offset += buf.length
    onProgress(offset / totalBytes)
  }
  if (offset !== totalBytes) throw new Error('Model download is incomplete')
  return out
}

async function getSession(req: Extract<WorkerRequest, { type: 'upscale' }>) {
  if (current?.id === req.model.id) return current
  await current?.session.release()
  current = undefined

  post({ type: 'progress', stage: `Loading model ${req.model.label}`, progress: 0 })
  const bytes = await fetchModel(req.model.parts, req.model.bytes, (f) => {
    post({ type: 'progress', stage: `Downloading model ${req.model.label}`, progress: f })
  })

  post({ type: 'progress', stage: 'Initialising GPU', progress: 0 })
  let session: ort.InferenceSession
  let backend: 'webgpu' | 'wasm' = 'webgpu'
  try {
    if (!('gpu' in navigator)) throw new Error('WebGPU unavailable')
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' })
  }
  catch (err) {
    console.warn('[upscaler] WebGPU unavailable, falling back to WASM:', err)
    backend = 'wasm'
    session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
  }
  current = { id: req.model.id, session, backend }
  post({ type: 'backend', backend })
  return current
}

/** Runs one 4× pass over an RGBA image using overlapping tiles. */
async function upscalePass(
  session: ort.InferenceSession,
  src: { width: number, height: number, data: Uint8ClampedArray },
  tile: number,
  onTile: () => void,
) {
  const S = 4
  const { width: w, height: h, data } = src
  const ow = w * S
  const oh = h * S
  const out = new Uint8ClampedArray(ow * oh * 4)

  for (let ty = 0; ty < h; ty += tile) {
    for (let tx = 0; tx < w; tx += tile) {
      // Tile bounds plus padding (to hide seams), clamped to the image.
      const x0 = Math.max(tx - TILE_PAD, 0)
      const y0 = Math.max(ty - TILE_PAD, 0)
      const x1 = Math.min(tx + tile + TILE_PAD, w)
      const y1 = Math.min(ty + tile + TILE_PAD, h)
      const tw = x1 - x0
      const th = y1 - y0
      const plane = tw * th

      const input = new Float32Array(3 * plane)
      for (let y = 0; y < th; y++) {
        let si = ((y0 + y) * w + x0) * 4
        let di = y * tw
        for (let x = 0; x < tw; x++, si += 4, di++) {
          input[di] = data[si]! / 255
          input[plane + di] = data[si + 1]! / 255
          input[2 * plane + di] = data[si + 2]! / 255
        }
      }

      const tensor = new ort.Tensor('float32', input, [1, 3, th, tw])
      const result = await session.run({ input: tensor })
      const output = result.output!
      const pixels = (await output.getData()) as Float32Array
      tensor.dispose()
      output.dispose()

      // Copy the un-padded centre of the tile into the output image.
      const otw = tw * S
      const oplane = otw * th * S
      const cx0 = (tx - x0) * S
      const cy0 = (ty - y0) * S
      const cw = (Math.min(tx + tile, w) - tx) * S
      const ch = (Math.min(ty + tile, h) - ty) * S
      for (let y = 0; y < ch; y++) {
        let si = (cy0 + y) * otw + cx0
        let di = ((ty * S + y) * ow + tx * S) * 4
        for (let x = 0; x < cw; x++, si++, di += 4) {
          out[di] = pixels[si]! * 255
          out[di + 1] = pixels[oplane + si]! * 255
          out[di + 2] = pixels[2 * oplane + si]! * 255
          out[di + 3] = 255
        }
      }
      onTile()
    }
  }

  // Upscale alpha with bilinear interpolation if the image has transparency.
  let hasAlpha = false
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] !== 255) { hasAlpha = true; break }
  }
  if (hasAlpha) {
    for (let y = 0; y < oh; y++) {
      const sy = Math.min(Math.max((y + 0.5) / S - 0.5, 0), h - 1)
      const y0 = Math.floor(sy)
      const y1 = Math.min(y0 + 1, h - 1)
      const fy = sy - y0
      for (let x = 0; x < ow; x++) {
        const sx = Math.min(Math.max((x + 0.5) / S - 0.5, 0), w - 1)
        const x0 = Math.floor(sx)
        const x1 = Math.min(x0 + 1, w - 1)
        const fx = sx - x0
        const a = data[(y0 * w + x0) * 4 + 3]! * (1 - fx) + data[(y0 * w + x1) * 4 + 3]! * fx
        const b = data[(y1 * w + x0) * 4 + 3]! * (1 - fx) + data[(y1 * w + x1) * 4 + 3]! * fx
        out[(y * ow + x) * 4 + 3] = a * (1 - fy) + b * fy
      }
    }
  }

  return { width: ow, height: oh, data: out }
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const req = e.data
  if (req.type !== 'upscale') return
  try {
    const { session, backend } = await getSession(req)
    const tile = backend === 'webgpu' ? req.model.tile : Math.min(req.model.tile, 128)

    let img = req.image
    // Count tiles across all passes for an overall progress figure.
    let totalTiles = 0
    let pw = img.width
    let ph = img.height
    for (let i = 0; i < req.passes; i++) {
      totalTiles += Math.ceil(pw / tile) * Math.ceil(ph / tile)
      pw *= 4
      ph *= 4
    }
    let doneTiles = 0
    for (let i = 0; i < req.passes; i++) {
      const stage = `AI upscaling${req.passes > 1 ? ` pass ${i + 1}/${req.passes}` : ''} (${backend === 'webgpu' ? 'WebGPU' : 'CPU'})`
      post({ type: 'progress', stage, progress: doneTiles / totalTiles })
      img = await upscalePass(session, img, tile, () => {
        doneTiles++
        post({ type: 'progress', stage, progress: doneTiles / totalTiles })
      })
    }
    post({ type: 'done', image: img, backend }, [img.data.buffer])
  }
  catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) })
  }
}
