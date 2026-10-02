<script setup lang="ts">
import manifest from '~~/public/models/manifest.json'
import type { ModelInfo } from '~/utils/upscaler-types'
import { MAX_DIMENSION, MAX_OUTPUT_PIXELS, planPasses, useUpscaler } from '~/composables/useUpscaler'
import type { Fit, OutputFormat, UpscaleResult } from '~/composables/useUpscaler'

const MAX_UPLOAD_MB = 50
const models: ModelInfo[] = manifest.models.map(m => ({ ...m, parts: m.parts.map(p => `/models/${p}`) }))
const { upscale: runUpscale, stage, progress } = useUpscaler()
const hasWebGPU = ref(true)
onMounted(() => { hasWebGPU.value = 'gpu' in navigator })

const PRESETS = [
  { label: '2×', factor: 2 },
  { label: '4×', factor: 4 },
  { label: '1080p', box: [1920, 1080] },
  { label: '1440p', box: [2560, 1440] },
  { label: '4K', box: [3840, 2160] },
  { label: '8K', box: [7680, 4320] },
] as const

const file = ref<File>()
const bitmap = shallowRef<ImageBitmap>()
const originalUrl = ref('')
const original = reactive({ width: 0, height: 0 })
const width = ref(0)
const height = ref(0)
const lockAspect = ref(true)
const activePreset = ref<string>('4×')
const fit = ref<Fit>('fill')
const model = ref(models[0]!.id)
const format = ref<OutputFormat>('png')
const quality = ref(92)
const dragging = ref(false)
const busy = ref(false)
const result = ref<UpscaleResult & { url: string, name: string, input: string, output: string }>()
const error = ref('')
const selectedModel = computed(() => models.find(m => m.id === model.value)!)
const passes = computed(() => (original.width ? planPasses(scaleFactor.value, original.width * original.height) : 0))

const aspect = computed(() => (original.width && original.height ? original.width / original.height : 1))
const scaleFactor = computed(() => (original.width ? Math.max(width.value / original.width, height.value / original.height) : 0))
const megapixels = computed(() => (width.value * height.value) / 1e6)
const tooLarge = computed(() => width.value * height.value > MAX_OUTPUT_PIXELS || width.value > MAX_DIMENSION || height.value > MAX_DIMENSION)
const aspectDiffers = computed(() => original.width && Math.abs(width.value / height.value - aspect.value) > 0.01)

function applyPreset(p: typeof PRESETS[number]) {
  activePreset.value = p.label
  if ('factor' in p) {
    width.value = Math.round(original.width * p.factor)
    height.value = Math.round(original.height * p.factor)
    return
  }
  const [bw, bh] = p.box
  if (lockAspect.value) {
    // Fit inside the box, keeping the original aspect ratio.
    const s = Math.min(bw / original.width, bh / original.height)
    width.value = Math.round(original.width * s)
    height.value = Math.round(original.height * s)
  }
  else {
    width.value = bw
    height.value = bh
  }
}

function onWidth(v: number) {
  activePreset.value = ''
  width.value = v
  if (lockAspect.value && v > 0) height.value = Math.max(1, Math.round(v / aspect.value))
}

function onHeight(v: number) {
  activePreset.value = ''
  height.value = v
  if (lockAspect.value && v > 0) width.value = Math.max(1, Math.round(v * aspect.value))
}

watch(lockAspect, (locked) => {
  if (locked && original.width) onWidth(width.value)
})

async function loadFile(f?: File | null) {
  error.value = ''
  if (!f) return
  if (!f.type.startsWith('image/')) {
    error.value = 'Please choose an image file.'
    return
  }
  if (f.size > MAX_UPLOAD_MB * 1024 * 1024) {
    error.value = `Image is larger than ${MAX_UPLOAD_MB} MB.`
    return
  }
  let bmp: ImageBitmap
  try {
    bmp = await createImageBitmap(f, { imageOrientation: 'from-image' })
  }
  catch {
    error.value = 'Could not read that image.'
    return
  }
  reset()
  file.value = f
  bitmap.value = bmp
  originalUrl.value = URL.createObjectURL(f)
  original.width = bmp.width
  original.height = bmp.height
  applyPreset(PRESETS.find(p => p.label === activePreset.value) ?? PRESETS[1])
}

function onDrop(e: DragEvent) {
  dragging.value = false
  loadFile(e.dataTransfer?.files?.[0])
}

function onPaste(e: ClipboardEvent) {
  const item = [...(e.clipboardData?.items ?? [])].find(i => i.type.startsWith('image/'))
  if (item) loadFile(item.getAsFile())
}

onMounted(() => window.addEventListener('paste', onPaste))
onBeforeUnmount(() => window.removeEventListener('paste', onPaste))

async function upscale() {
  if (!bitmap.value) return
  error.value = ''
  clearResult()
  busy.value = true
  stage.value = 'Starting'
  progress.value = 0
  try {
    const r = await runUpscale({
      source: bitmap.value,
      model: selectedModel.value,
      width: width.value,
      height: height.value,
      fit: fit.value,
      format: format.value,
      quality: quality.value,
    })
    const base = file.value!.name.replace(/\.[^.]+$/, '').replace(/[^\w.-]+/g, '_') || 'image'
    result.value = {
      ...r,
      url: URL.createObjectURL(r.blob),
      name: `${base}_${width.value}x${height.value}.${format.value === 'jpeg' ? 'jpg' : format.value}`,
      input: `${original.width}×${original.height}`,
      output: `${width.value}×${height.value}`,
    }
  }
  catch (e) {
    console.error(e)
    error.value = e instanceof Error ? e.message : 'Upscaling failed.'
  }
  finally {
    busy.value = false
  }
}

function clearResult() {
  if (result.value) URL.revokeObjectURL(result.value.url)
  result.value = undefined
}

function reset() {
  clearResult()
  if (originalUrl.value) URL.revokeObjectURL(originalUrl.value)
  bitmap.value?.close()
  bitmap.value = undefined
  file.value = undefined
  originalUrl.value = ''
  error.value = ''
}
</script>

<template>
  <div class="page">
    <header>
      <h1>Real-ESRGAN Upscaler</h1>
      <p>Upscale images with AI to any resolution. Runs entirely in your browser — your images never leave your device.</p>
    </header>

    <div v-if="!hasWebGPU" class="banner">
      WebGPU isn't available in this browser, so upscaling will run on the CPU and be much slower.
      For best results use a recent Chrome, Edge or Safari.
    </div>

    <main class="layout">
      <section class="stage">
        <label
          v-if="!file"
          class="drop"
          :class="{ dragging }"
          @dragover.prevent="dragging = true"
          @dragleave="dragging = false"
          @drop.prevent="onDrop"
        >
          <input type="file" accept="image/png,image/jpeg,image/webp" hidden @change="loadFile(($event.target as HTMLInputElement).files?.[0])">
          <strong>Drop an image here</strong>
          <span>or click to browse · paste from clipboard · PNG, JPEG, WebP up to {{ MAX_UPLOAD_MB }} MB</span>
        </label>

        <template v-else>
          <CompareSlider v-if="result" :before="originalUrl" :after="result.url" :fit="fit" />
          <div v-else class="preview">
            <img :src="originalUrl" alt="Original image">
          </div>

          <div class="meta">
            <span>{{ file.name }} · {{ original.width }}×{{ original.height }}</span>
            <button class="link" :disabled="busy" @click="reset">Choose another image</button>
          </div>
        </template>
      </section>

      <aside class="panel">
        <fieldset :disabled="!file || busy">
          <legend>Target resolution</legend>
          <div class="chips">
            <button
              v-for="p in PRESETS"
              :key="p.label"
              type="button"
              :class="{ active: activePreset === p.label }"
              @click="applyPreset(p)"
            >
              {{ p.label }}
            </button>
          </div>

          <div class="dims">
            <label>Width
              <input type="number" min="1" max="20000" :value="width" @input="onWidth(+($event.target as HTMLInputElement).value)">
            </label>
            <button
              type="button"
              class="lock"
              :class="{ active: lockAspect }"
              :aria-pressed="lockAspect"
              :title="lockAspect ? 'Aspect ratio locked' : 'Aspect ratio unlocked'"
              @click="lockAspect = !lockAspect"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <rect x="5" y="11" width="14" height="10" rx="2" />
                <path :d="lockAspect ? 'M8 11V7a4 4 0 0 1 8 0v4' : 'M8 11V7a4 4 0 0 1 7.5-2'" />
              </svg>
            </button>
            <label>Height
              <input type="number" min="1" max="20000" :value="height" @input="onHeight(+($event.target as HTMLInputElement).value)">
            </label>
          </div>

          <p class="hint" :class="{ warn: tooLarge }">
            {{ megapixels.toFixed(1) }} MP · {{ scaleFactor.toFixed(2) }}× scale
            · {{ passes ? `${passes} AI pass${passes > 1 ? 'es' : ''}` : 'no AI (downscale)' }}
            <template v-if="tooLarge"> · exceeds {{ MAX_OUTPUT_PIXELS / 1e6 }} MP / {{ MAX_DIMENSION }} px limit</template>
          </p>

          <label v-if="aspectDiffers">Aspect ratio mismatch
            <select v-model="fit">
              <option value="fill">Stretch to fit</option>
              <option value="cover">Crop to fill</option>
              <option value="contain">Pad (letterbox)</option>
            </select>
          </label>
        </fieldset>

        <fieldset :disabled="!file || busy">
          <legend>Model</legend>
          <label v-for="m in models" :key="m.id" class="radio">
            <input v-model="model" type="radio" name="model" :value="m.id">
            <span>
              <strong>{{ m.label }}</strong>
              <small>{{ m.description }} · {{ (m.bytes / 1e6).toFixed(0) }} MB download</small>
            </span>
          </label>
        </fieldset>

        <fieldset :disabled="!file || busy">
          <legend>Output</legend>
          <div class="row">
            <label>Format
              <select v-model="format">
                <option value="png">PNG</option>
                <option value="jpeg">JPEG</option>
                <option value="webp">WebP</option>
              </select>
            </label>
            <label v-if="format !== 'png'">Quality
              <input v-model.number="quality" type="number" min="1" max="100">
            </label>
          </div>
        </fieldset>

        <button class="primary" :disabled="!file || busy || tooLarge || width < 1 || height < 1" @click="upscale">
          {{ busy ? 'Upscaling…' : 'Upscale' }}
        </button>

        <div v-if="busy" class="progress" role="status">
          <div class="bar"><div :style="{ width: `${Math.round(progress * 100)}%` }" /></div>
          <small>{{ stage }} · {{ Math.round(progress * 100) }}%</small>
        </div>

        <p v-if="error" class="error" role="alert">{{ error }}</p>

        <div v-if="result" class="result">
          <p>
            {{ result.input }} → <strong>{{ result.output }}</strong><br>
            <small>
              AI {{ result.passes ? `${4 ** result.passes}× on ${result.backend === 'webgpu' ? 'WebGPU' : 'CPU'}` : 'skipped (downscale)' }}
              · {{ result.seconds }}s · {{ (result.blob.size / 1e6).toFixed(1) }} MB
            </small>
          </p>
          <a class="primary" :href="result.url" :download="result.name">Download</a>
        </div>
      </aside>
    </main>
  </div>
</template>

<style scoped>
.page {
  max-width: 1280px;
  margin: 0 auto;
  padding: 24px 16px 48px;
}
header h1 { margin: 0; font-size: 26px; }
header p { margin: 4px 0 20px; color: var(--muted); }

.banner {
  margin-bottom: 16px;
  padding: 12px 16px;
  border: 1px solid var(--danger);
  border-radius: var(--radius);
  color: var(--danger);
}

.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: 20px;
  align-items: start;
}
@media (max-width: 860px) {
  .layout { grid-template-columns: 1fr; }
}

.stage {
  min-width: 0;
  text-align: center;
}

.drop {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-height: 420px;
  padding: 24px;
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  color: var(--muted);
  cursor: pointer;
  transition: border-color .15s, background .15s;
}
.drop strong { color: var(--text); font-size: 18px; }
.drop:hover, .drop.dragging { border-color: var(--accent); }

.preview {
  border-radius: var(--radius);
  overflow: hidden;
  background: repeating-conic-gradient(var(--checker) 0 25%, transparent 0 50%) 0 0 / 20px 20px;
}
.preview img {
  display: block;
  max-width: 100%;
  max-height: 70vh;
  margin: 0 auto;
}

.meta {
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
  color: var(--muted);
  font-size: 14px;
  text-align: left;
}

.link {
  padding: 0;
  border: 0;
  background: none;
  color: var(--accent);
}

.panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

fieldset {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin: 0;
  padding: 0;
  border: 0;
}
fieldset:disabled { opacity: .6; }
legend {
  margin-bottom: 8px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: .06em;
  text-transform: uppercase;
  color: var(--muted);
}

label { display: flex; flex-direction: column; gap: 4px; font-size: 13px; color: var(--muted); }

.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chips button {
  padding: 5px 12px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  font-size: 13px;
}
.chips button.active {
  border-color: var(--accent);
  background: var(--accent);
  color: var(--accent-contrast);
}

.dims {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 8px;
  align-items: end;
}
.lock {
  display: grid;
  place-items: center;
  width: 36px;
  height: 38px;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--surface-2);
  color: var(--muted);
}
.lock.active { color: var(--accent); border-color: var(--accent); }

.row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

.hint { margin: 0; font-size: 13px; color: var(--muted); }
.hint.warn { color: var(--danger); }

.radio, .check {
  flex-direction: row;
  align-items: flex-start;
  gap: 10px;
  color: var(--text);
  font-size: 14px;
  cursor: pointer;
}
.radio input, .check input { margin-top: 4px; accent-color: var(--accent); }
.radio span { display: flex; flex-direction: column; }
small { color: var(--muted); font-size: 12px; }

.primary {
  display: block;
  padding: 11px;
  border: 0;
  border-radius: 10px;
  background: var(--accent);
  color: var(--accent-contrast);
  font-weight: 600;
  text-align: center;
  text-decoration: none;
}

.progress { display: flex; flex-direction: column; gap: 6px; }
.bar { height: 8px; border-radius: 4px; background: var(--surface-2); overflow: hidden; }
.bar div { height: 100%; background: var(--accent); transition: width .3s; }

.error { margin: 0; color: var(--danger); font-size: 14px; }

.result p { margin: 0 0 10px; }
</style>
