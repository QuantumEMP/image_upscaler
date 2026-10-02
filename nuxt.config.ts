import { fileURLToPath } from 'node:url'

// Cross-origin isolation enables multi-threaded WASM (used when WebGPU is unavailable).
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

export default defineNuxtConfig({
  compatibilityDate: '2026-10-01',
  devtools: { enabled: true },
  app: {
    head: {
      title: 'Real-ESRGAN Upscaler',
      meta: [
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
        { name: 'description', content: 'Upscale images to any resolution with Real-ESRGAN, entirely in your browser.' },
      ],
    },
  },
  css: ['~/assets/main.css'],
  // Fully static site: `nuxt generate` output is served by Cloudflare Workers static assets.
  // Production headers live in public/_headers; these mirror them for `nuxt dev`.
  // Pin the preset: in Cloudflare's CI Nitro would otherwise auto-select `cloudflare-module`, which
  // redirects wrangler to a generated config whose Worker entry (index.mjs) a static build never emits.
  nitro: { preset: 'static' },
  routeRules: {
    '/**': { headers: isolationHeaders },
  },
  vite: {
    server: { headers: isolationHeaders },
    resolve: {
      // Use the ORT build that doesn't bundle its 26.8 MB wasm (over Cloudflare's 25 MiB per-file limit);
      // the worker loads the runtime from jsDelivr instead.
      alias: [{ find: /^onnxruntime-web\/webgpu$/, replacement: fileURLToPath(new URL('./node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs', import.meta.url)) }],
    },
    optimizeDeps: { exclude: ['onnxruntime-web'] },
    worker: { format: 'es' },
  },
})
