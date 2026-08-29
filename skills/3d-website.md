---
name: 3d-website
triggers: 3d website, 3d site, 3d web, 3d landing, 3d portfolio, 3d experience, 3d scene, three.js, threejs, babylon.js, babylonjs, webgl site, webgl website, webgl experience
---
3D WEBSITE SKILL — apply only because this request is for a 3D / WebGL / Three.js / immersive website.

Framework choice:
- Default to Three.js (MIT, load via ES module import from a CDN like unpkg/jsdelivr) — most flexible and best-supported option for a single-page/static build.
- Reach for Babylon.js only if the request explicitly needs built-in physics, a full scene GUI, or heavy built-in post-processing.
- Do not pull in a full game engine (Unity WebGL) or a proprietary SaaS embed (Spline) unless the user explicitly asks for it — both add unnecessary weight or lock-in for a website.
- Pair Three.js with GSAP + ScrollTrigger for choreographed motion, and native IntersectionObserver for simple reveals — same libraries already called for in the motion-system rules above.

Assets — there is no art pipeline (no Blender/Substance/Draco/KTX2) available here, so:
- Prefer procedural/primitive geometry (spheres, boxes, torus, custom BufferGeometry, particle systems, extrusions) and code-driven materials/shaders over requiring model files that don't exist.
- If a real model is genuinely needed and none is supplied, say so and offer a primitive-based placeholder — never fabricate a fake model URL or silently assume an asset exists.
- If the user does supply or link a model, only rely on glTF/GLB (the open, PBR-capable, web-native format) — never expect FBX/OBJ/USD to just work in-browser without conversion.
- Keep triangle counts and texture sizes modest by construction (procedural geometry, small canvas-generated or CDN placeholder textures), since there's no runtime compression step available.

Cinematic design & motion — apply the visual-language and motion-system rules above, tuned for 3D:
- Restrained, low-to-medium saturation palette; soft directional lighting over flat ambient; avoid neon/garish colors.
- Compose with rule-of-thirds and off-center focal objects; use foreground elements for parallax depth.
- Camera moves must be slow and purposeful — gentle pans/dolly, never a sudden jolt or uncontrolled shake.
- Ease-out on entrances, ease-in on exits; major transitions ~300-600ms; avoid perfectly linear motion.
- Give moving objects anticipation and a slight overshoot/settle rather than a flat linear tween.
- Small idle motion (gentle float/breathing) is fine; constant busy motion everywhere is not.

Performance, accessibility, SEO — non-negotiable for any 3D build:
- Render the real headline, body copy, nav, and CTAs as actual HTML first, outside/behind the canvas — the WebGL layer is a progressive enhancement, not the only source of content. Crawlers and non-JS clients see nothing inside a <canvas>.
- Put a <noscript> fallback with the core headline/summary/CTA inside the canvas container.
- Label the <canvas> with aria-label/role, and keep primary navigation and calls-to-action as normal HTML controls, never 3D-only interactions.
- Respect prefers-reduced-motion: reduce — disable or simplify camera moves, parallax, and non-essential animation loops (same reduced-motion rule already required above).
- Cap devicePixelRatio (Math.min(devicePixelRatio, 2)), pause the render loop when the canvas is off-screen or the tab is hidden, and clean up the renderer/geometries/materials/textures plus any requestAnimationFrame/ScrollTrigger instances on teardown (same WebGL performance and animation-lifecycle rules already required above).
- On low-power devices or very small viewports, prefer a lighter fallback (fewer particles, lower-res canvas, or a static hero image) over forcing the full scene — 3D must never block the page from being usable.
