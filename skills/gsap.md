---
name: gsap
triggers: gsap, greensock, scrolltrigger, scroll-driven animation, scroll animation, staggered reveal, timeline animation
---
GSAP SKILL — use for high-quality UI motion: entrances, micro-interactions, timeline sequencing, and scroll-driven storytelling. Load via CDN script tags (`https://cdn.jsdelivr.net/npm/gsap@3/dist/gsap.min.js` and, if scroll effects are needed, `.../ScrollTrigger.min.js`), then `gsap.registerPlugin(ScrollTrigger)`.

Key APIs: `gsap.to/from/fromTo(targets, vars)` for single tweens; `gsap.timeline({ defaults, repeat, yoyo })` for sequences, chained with position params (absolute `1.2`, relative `"+=0.5"`, overlap `"-=0.3"`, or a label). Eases like `"power2.out"`, `"expo.inOut"`, `"elastic.out(1, 0.3)"`. Stagger with `stagger: 0.05` or `{ each, from: "center" }`.

Prefer animating transforms (`x`, `y`, `scale`, `rotation`) and `autoAlpha` over layout properties (top/left/width/height) — animating layout causes jank. Add `will-change: transform` on animated elements.

Recipes:
```js
// Hero entrance stagger
gsap.from(".hero [data-anim]", { y: 24, autoAlpha: 0, duration: 0.8, ease: "power2.out", stagger: 0.06 });

// Sequenced timeline
const tl = gsap.timeline({ defaults: { ease: "power2.out", duration: 0.6 } });
tl.from(".nav", { y: -20, autoAlpha: 0 })
  .from(".hero-title", { y: 30, autoAlpha: 0 }, "-=0.2")
  .from(".hero-cta", { scale: 0.95, autoAlpha: 0 }, "-=0.2");

// Scroll-scrubbed pinned section
gsap.timeline({ scrollTrigger: { trigger: ".story", start: "top top", end: "+=800", scrub: 1, pin: true } })
  .to(".story .panel", { xPercent: -200 });
```

Pitfalls: ScrollTrigger "not firing" is usually a trigger with no height, or a scroll container other than the window that wasn't configured; run `ScrollTrigger.refresh()` after images/fonts finish loading so measurements are correct; respect `prefers-reduced-motion` by skipping scrub/pin timelines and rendering the final state immediately; only ever run one smooth-scroll engine (Lenis or similar) alongside ScrollTrigger, never both, and destroy it on teardown.
