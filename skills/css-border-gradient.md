---
name: css-border-gradient
triggers: gradient border, border gradient, glowing edge, premium card edge
---
CSS GRADIENT BORDER SKILL — a refined edge highlight for premium surfaces (hero panels, pricing cards, modals), without a loud glow.

Defaults: 1px width (2px only for large hero cards), radius inherits the parent, angle 135deg-160deg, keep most color stops below 0.4 opacity — subtle beats shiny.

Simple pattern (solid/translucent fill):
```css
.gradient-border {
  --surface: rgba(10, 14, 24, 0.72);
  border: 1px solid transparent;
  border-radius: 20px;
  background:
    linear-gradient(var(--surface), var(--surface)) padding-box,
    linear-gradient(135deg, rgba(255,255,255,0.34), rgba(125,92,255,0.36), rgba(255,255,255,0.08)) border-box;
}
```

Masked pattern (when the surface already has a complex background that must not be overwritten):
```css
.gradient-border-mask { position: relative; border-radius: 20px; }
.gradient-border-mask::before {
  content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 1px;
  background: linear-gradient(145deg, rgba(255,255,255,0.34), rgba(125,92,255,0.36) 45%, rgba(255,255,255,0.08));
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
}
```

Taste rules: apply to one hierarchy level at a time (primary card, active tab, selected plan) — not everywhere; never use rainbow or full-saturation neon borders; the border should frame the content, not compete with it; check light and dark themes separately, since the same alpha rarely reads correctly on both.
