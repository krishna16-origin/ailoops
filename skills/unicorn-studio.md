---
name: unicorn-studio
triggers: unicorn studio, unicornstudio
---
UNICORN STUDIO SKILL — embedding a designed, no-code WebGL scene (built in the Unicorn Studio editor) rather than hand-coding shaders. Requires a project ID from the user; never invent one.

```html
<div style="width: 100%; height: 420px" data-us-project="USER_PROVIDED_PROJECT_ID"></div>
<script src="https://cdn.jsdelivr.net/gh/hiunicornstudio/unicornstudio.js/dist/unicornStudio.umd.js"></script>
<script>UnicornStudio.init();</script>
```

Performance-first embed adds attributes to the container: `data-us-lazyload="true"`, `data-us-scale="0.75"` (render scale), `data-us-dpi="1.25"`, `data-us-fps="45"`.

Pitfalls: the container needs explicit width/height or the scene won't display; keep to well under 10 scenes on one page (WebGL context limits, roughly 16 max in a browser); reduce scale/dpi/fps on low-end devices instead of dropping the effect entirely.
