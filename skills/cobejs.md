---
name: cobejs
triggers: cobe.js, cobe globe, lightweight globe, spinning globe markers
---
COBE.JS SKILL — a small, focused WebGL globe (not full Three.js) for a "spinning globe with markers" hero or about-page moment. Load via ESM CDN import:

```html
<canvas id="cobe" style="width: 100%; height: 100%; max-width: 500px; aspect-ratio: 1;"></canvas>
<script type="module">
import createGlobe from "https://cdn.jsdelivr.net/npm/cobe/dist/index.js";

const canvas = document.getElementById("cobe");
let phi = 0;

function setup() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio, 2);
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const globe = createGlobe(canvas, {
    devicePixelRatio: dpr, width: canvas.width, height: canvas.height,
    phi: 0, theta: 0.2, dark: 0, diffuse: 1.2, scale: 1,
    mapSamples: 16000, mapBrightness: 6,
    baseColor: [0.2, 0.2, 0.25], glowColor: [1, 1, 1], markerColor: [0.8, 0.5, 1],
    markers: [{ location: [37.7749, -122.4194], size: 0.08 }],
    onRender: (state) => { state.phi = phi; phi += 0.01; },
  });
  return globe;
}

let globe = setup();
window.addEventListener("resize", () => { globe.destroy(); globe = setup(); });
</script>
```

Set the canvas's CSS size AND its actual `width`/`height` attributes scaled by `devicePixelRatio` (clamped to 2) — a mismatch here is the most common bug. `globe.toggle()` pauses the render loop; `globe.destroy()` tears it down before removing the canvas or re-initializing on resize.
