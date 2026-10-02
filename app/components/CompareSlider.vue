<script setup lang="ts">
defineProps<{ before: string, after: string, fit: 'fill' | 'cover' | 'contain' }>()

const pos = ref(50)
const el = ref<HTMLElement>()

function update(e: PointerEvent) {
  const rect = el.value!.getBoundingClientRect()
  pos.value = Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100))
}

function onDown(e: PointerEvent) {
  el.value!.setPointerCapture(e.pointerId)
  update(e)
}

function onKey(e: KeyboardEvent) {
  if (e.key === 'ArrowLeft') pos.value = Math.max(0, pos.value - 5)
  if (e.key === 'ArrowRight') pos.value = Math.min(100, pos.value + 5)
}
</script>

<template>
  <div
    ref="el"
    class="compare"
    role="slider"
    tabindex="0"
    aria-label="Before / after comparison"
    :aria-valuenow="Math.round(pos)"
    aria-valuemin="0"
    aria-valuemax="100"
    @pointerdown="onDown"
    @pointermove="(e) => e.buttons && update(e)"
    @keydown="onKey"
  >
    <img :src="after" alt="Upscaled" draggable="false">
    <img :src="before" alt="Original" class="before" :style="{ objectFit: fit, clipPath: `inset(0 ${100 - pos}% 0 0)` }" draggable="false">
    <div class="handle" :style="{ left: `${pos}%` }"><span /></div>
    <span class="tag left">Original</span>
    <span class="tag right">Real-ESRGAN</span>
  </div>
</template>

<style scoped>
.compare {
  position: relative;
  display: inline-block;
  max-width: 100%;
  vertical-align: top;
  overflow: hidden;
  border-radius: var(--radius);
  cursor: ew-resize;
  user-select: none;
  touch-action: none;
  background: repeating-conic-gradient(var(--checker) 0 25%, transparent 0 50%) 0 0 / 20px 20px;
}
img {
  display: block;
  max-width: 100%;
  max-height: 70vh;
}
.before {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  image-rendering: pixelated;
}
.handle {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #fff;
  box-shadow: 0 0 6px rgb(0 0 0 / .5);
  transform: translateX(-1px);
}
.handle span {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: #fff;
  transform: translate(-50%, -50%);
  box-shadow: 0 1px 6px rgb(0 0 0 / .4);
}
.tag {
  position: absolute;
  top: 10px;
  padding: 2px 8px;
  font-size: 12px;
  color: #fff;
  background: rgb(0 0 0 / .55);
  border-radius: 6px;
}
.left { left: 10px; }
.right { right: 10px; }
</style>
