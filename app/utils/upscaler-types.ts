export interface ModelInfo {
  id: string
  label: string
  description: string
  scale: number
  tile: number
  bytes: number
  parts: string[]
}

export interface RawImage {
  width: number
  height: number
  data: Uint8ClampedArray
}

export type WorkerRequest =
  | { type: 'upscale', model: ModelInfo, image: RawImage, passes: number }

export type WorkerResponse =
  | { type: 'progress', stage: string, progress: number }
  | { type: 'backend', backend: 'webgpu' | 'wasm' }
  | { type: 'done', image: RawImage, backend: 'webgpu' | 'wasm' }
  | { type: 'error', message: string }
