import type { ModelInfo, RawImage, WorkerRequest, WorkerResponse } from '~/utils/upscaler-types'

export type Fit = 'fill' | 'cover' | 'contain'
export type OutputFormat = 'png' | 'jpeg' | 'webp'

export interface UpscaleOptions {
  source: ImageBitmap
  model: ModelInfo
  width: number
  height: number
  fit: Fit
  format: OutputFormat
  quality: number
}

export interface UpscaleResult {
  blob: Blob
  passes: number
  backend: 'webgpu' | 'wasm' | 'none'
  seconds: number
}

// Limits keep memory and canvas sizes within what browsers reliably support.
export const MAX_OUTPUT_PIXELS = 64_000_000
export const MAX_DIMENSION = 16_384
const MAX_INTERMEDIATE_PIXELS = 64_000_000
const EXTRA_PASS_THRESHOLD = 1.5

/** Number of 4× AI passes needed to reach `ratio` (small remainders are left to Lanczos). */
export function planPasses(ratio: number, inputPixels: number) {
  let passes = 0
  let current = 1
  while (ratio / current > (passes ? EXTRA_PASS_THRESHOLD : 1)) {
    if (inputPixels * (current * 4) ** 2 > MAX_INTERMEDIATE_PIXELS) break
    passes++
    current *= 4
  }
  return passes
}

let picaInstance: Promise<import('pica').Pica> | undefined
function getPica() {
  picaInstance ??= import('pica').then(({ Pica }) => new Pica({ features: ['js', 'wasm', 'ww'] }))
  return picaInstance
}

function makeCanvas(width: number, height: number) {
  const c = document.createElement('canvas')
  c.width = width
  c.height = height
  return c
}

export function useUpscaler() {
  const stage = ref('')
  const progress = ref(0)
  const backend = ref<'webgpu' | 'wasm'>()
  let worker: Worker | undefined

  function getWorker() {
    worker ??= new Worker(new URL('../workers/upscaler.worker.ts', import.meta.url), { type: 'module' })
    return worker
  }

  function runWorker(req: WorkerRequest, transfer: Transferable[]) {
    return new Promise<{ image: RawImage, backend: 'webgpu' | 'wasm' }>((resolve, reject) => {
      const w = getWorker()
      w.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const msg = e.data
        if (msg.type === 'progress') {
          stage.value = msg.stage
          progress.value = msg.progress
        }
        else if (msg.type === 'backend') backend.value = msg.backend
        else if (msg.type === 'done') resolve(msg)
        else if (msg.type === 'error') reject(new Error(msg.message))
      }
      w.onerror = (e) => {
        worker?.terminate()
        worker = undefined
        reject(new Error(e.message || 'Upscaler worker crashed'))
      }
      w.postMessage(req, transfer)
    })
  }

  async function upscale(opts: UpscaleOptions): Promise<UpscaleResult> {
    const started = performance.now()
    const { source, width, height, fit } = opts
    const ratio = Math.max(width / source.width, height / source.height)
    const passes = planPasses(ratio, source.width * source.height)

    let canvas = makeCanvas(source.width, source.height)
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(source, 0, 0)
    let usedBackend: UpscaleResult['backend'] = 'none'

    if (passes > 0) {
      const imageData = ctx.getImageData(0, 0, source.width, source.height)
      const raw: RawImage = { width: imageData.width, height: imageData.height, data: imageData.data }
      const { image, backend: b } = await runWorker(
        { type: 'upscale', model: toRaw(opts.model), image: raw, passes },
        [raw.data.buffer],
      )
      usedBackend = b
      canvas = makeCanvas(image.width, image.height)
      canvas.getContext('2d')!.putImageData(new ImageData(image.data as Uint8ClampedArray<ArrayBuffer>, image.width, image.height), 0, 0)
    }

    stage.value = 'Resizing to target resolution'
    progress.value = 1

    // Work out source crop (cover) and destination rect (contain).
    let sx = 0, sy = 0, sw = canvas.width, sh = canvas.height
    let dw = width, dh = height
    if (fit === 'cover') {
      const s = Math.max(width / sw, height / sh)
      const cw = Math.round(width / s)
      const ch = Math.round(height / s)
      sx = Math.floor((sw - cw) / 2)
      sy = Math.floor((sh - ch) / 2)
      sw = cw
      sh = ch
    }
    else if (fit === 'contain') {
      const s = Math.min(width / sw, height / sh)
      dw = Math.max(1, Math.round(sw * s))
      dh = Math.max(1, Math.round(sh * s))
    }

    let src = canvas
    if (sx || sy || sw !== canvas.width || sh !== canvas.height) {
      src = makeCanvas(sw, sh)
      src.getContext('2d')!.drawImage(canvas, sx, sy, sw, sh, 0, 0, sw, sh)
    }

    const pica = await getPica()
    const resized = makeCanvas(dw, dh)
    await pica.resize(src, resized, { filter: 'lanczos3' })

    let final = resized
    if (dw !== width || dh !== height || opts.format === 'jpeg') {
      final = makeCanvas(width, height)
      const fctx = final.getContext('2d')!
      if (opts.format === 'jpeg') {
        fctx.fillStyle = '#000'
        fctx.fillRect(0, 0, width, height)
      }
      fctx.drawImage(resized, Math.floor((width - dw) / 2), Math.floor((height - dh) / 2))
    }

    stage.value = 'Encoding'
    const blob = await new Promise<Blob>((resolve, reject) =>
      final.toBlob(b => (b ? resolve(b) : reject(new Error('Encoding failed — image may be too large for this browser'))), `image/${opts.format}`, opts.quality / 100),
    )
    if (blob.type !== `image/${opts.format}`) throw new Error(`This browser cannot encode ${opts.format.toUpperCase()}`)

    return { blob, passes, backend: usedBackend, seconds: Math.round((performance.now() - started) / 100) / 10 }
  }

  onBeforeUnmount(() => worker?.terminate())

  return { upscale, stage, progress, backend }
}
