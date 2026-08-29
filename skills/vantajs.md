---
name: vantajs
triggers: vanta.js, vantajs, animated background waves, animated webgl background
---
VANTA.JS SKILL — a quick animated WebGL background behind a hero section, without hand-building a Three.js scene. Load three.js first, then one Vanta effect bundle, via CDN:

```html
<div id="hero" style="height: 70vh;"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vanta/dist/vanta.waves.min.js"></script>
<script>
  const effect = VANTA.WAVES({ el: "#hero", color: 0x0b1220, shininess: 40, waveHeight: 16, zoom: 0.9 });
</script>
```

Other effects follow the same pattern: swap the bundle filename (`vanta.fog.min.js`, `vanta.net.min.js`, `vanta.birds.min.js`, etc.) and the global (`VANTA.FOG`, `VANTA.NET`, `VANTA.BIRDS`).

Pitfalls: the target element needs an explicit height or it renders nothing; keep to 1-2 WebGL effects per page (GPU cost adds up); if the effect sits behind text, verify contrast/readability at the chosen color/zoom; call `effect.destroy()` before removing the container, and `effect.resize()` if the container's size changes after init.
