---
name: css-alpha-masking
triggers: alpha mask, mask gradient, fade edge, edge fade, css mask-image
---
CSS ALPHA MASKING SKILL — fade an element's edges using `mask-image`/`-webkit-mask-image` with a linear gradient (always include the `-webkit-` prefix for Safari).

Horizontal fade (left/right):
```css
mask-image: linear-gradient(to right, transparent, black 15%, black 85%, transparent);
-webkit-mask-image: linear-gradient(to right, transparent, black 15%, black 85%, transparent);
```

Vertical fade (top/bottom):
```css
mask-image: linear-gradient(to bottom, transparent, black 15%, black 85%, transparent);
-webkit-mask-image: linear-gradient(to bottom, transparent, black 15%, black 85%, transparent);
```

Tune the `15%`/`85%` stops to widen or narrow the fade, and swap `transparent` for a low-alpha `rgba(0,0,0,0.2)` for a softer edge. The masked element needs actual visible content behind it — the mask only reveals/hides existing pixels, it doesn't create a background.
