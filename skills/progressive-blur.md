---
name: progressive-blur
triggers: progressive blur, gradient blur, layered blur, stepped blur
---
PROGRESSIVE BLUR SKILL — a smooth blur falloff from an edge of the viewport (e.g. content scrolling under a translucent top/bottom bar). `backdrop-filter: blur()` can't itself be gradiented, so fake it by stacking several full-size layers, each blurred more than the last and masked to a narrower band, so they overlap into a smooth gradient.

Structure: a fixed-position container (`position: fixed; inset: 0 0 auto 0` for a top blur, or `inset: auto 0 0 0` for a bottom blur; `pointer-events: none`) holding N stacked full-size layers (use `::before`, several `<div>`s, and `::after` for convenience). Layer *i* (0-indexed) gets:
- `backdrop-filter: blur(2^i * 0.5px)` — each layer roughly doubles the blur of the previous one (0.5px, 1px, 2px, 4px, 8px, 16px, 32px, 64px for an 8-layer stack).
- a `mask: linear-gradient(to <edge-direction>, transparent A%, black B%, black C%, transparent D%)` where the opaque window slides further from the edge as *i* increases, so each layer only shows its blur in its own band and the bands overlap smoothly.

```css
.gradient-blur { position: fixed; z-index: 5; inset: 0 0 auto 0; height: 12%; pointer-events: none; }
.gradient-blur > div, .gradient-blur::before, .gradient-blur::after { position: absolute; inset: 0; }
/* Repeat for each layer, doubling backdrop-filter blur and sliding the mask window one step further from the edge each time. */
.gradient-blur::before { backdrop-filter: blur(0.5px); mask: linear-gradient(to top, transparent 0%, black 12.5%, black 25%, transparent 37.5%); }
.gradient-blur > div:nth-of-type(1) { backdrop-filter: blur(1px); mask: linear-gradient(to top, transparent 12.5%, black 25%, black 37.5%, transparent 50%); }
```

For a bottom blur, flip `inset` to anchor at the bottom and change `linear-gradient(to top, ...)` to `linear-gradient(to bottom, ...)` in every layer.

Usage checklist: place the container near the top of `<body>`; make sure real content exists behind it (backdrop-filter blurs what's behind, it doesn't create anything); set `z-index` above page content but below any modal; adjust `height`/number of layers to taste (more layers = smoother falloff, at a small perf cost).
