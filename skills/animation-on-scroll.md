---
name: animation-on-scroll
triggers: animate on scroll, scroll reveal, reveal on scroll, fade in on scroll, scroll-triggered animation
---
ANIMATION ON SCROLL SKILL — lightweight reveal-on-scroll using IntersectionObserver, no animation library required. Insert once in `<head>` or before `</body>`:

```html
<style>
  @keyframes revealIn { 0% { opacity: 0; transform: translateY(30px); filter: blur(8px); } 100% { opacity: 1; transform: translateY(0); filter: blur(0); } }
  .animate-on-scroll { animation: revealIn 0.8s ease-out 0.1s both; animation-play-state: paused; }
  .animate-on-scroll.in-view { animation-play-state: running; }
</style>
<script>
  (function () {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          io.unobserve(entry.target); // remove this line to let it replay on re-entry
        }
      });
    }, { threshold: 0.2, rootMargin: "0px 0px -10% 0px" });
    document.addEventListener("DOMContentLoaded", () => {
      document.querySelectorAll(".animate-on-scroll").forEach((el) => io.observe(el));
    });
  })();
</script>
```

Apply `class="animate-on-scroll"` to any element that should reveal on entry. Tune `threshold`/`rootMargin` for earlier or later reveals, and the keyframe's translateY/blur values for a subtler or bolder motion. Respect `prefers-reduced-motion: reduce` by skipping the animation and showing the final state immediately.

Pitfall: an element already in the viewport before the observer initializes needs its class applied on load, not just on scroll — check visibility once at `DOMContentLoaded` if above-the-fold content also needs the effect.
