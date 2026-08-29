# Code Mode skills

Each `.md` file here is one skill: a small frontmatter block plus condensed,
actionable content. `app.py` loads every file in this folder, and — only when
the user's latest message contains one of a skill's `triggers` — injects that
skill's body into the Code Mode system prompt for that turn. No skill is ever
injected into Chat Mode, and a skill that doesn't match anything costs zero
tokens.

## Format

```markdown
---
name: gsap
triggers: gsap, scrolltrigger, scroll animation, staggered reveal
---
<the actual skill content, in your own words, that the model can act on>
```

- `name` — used for logging/identification, defaults to the filename if omitted.
- `triggers` — comma-separated phrases, matched as case-insensitive substrings
  against the user's latest message. Prefer multi-word phrases or specific
  library names (`"three.js"`, `"gradient border"`) over single common words
  (`"3d"`, `"scroll"`) so unrelated requests don't accidentally match.
- Body — whatever the model needs to actually act on the request: setup
  snippets, pitfalls, defaults, taste rules. Keep it to what an agent writing
  code in a single pass can use — this app has no build step, no package
  manager, and no art pipeline, so every recipe here loads from a CDN and
  needs no `npm install`.

## Adding a new skill or library

Drop a new `.md` file in this folder following the format above — no code
changes needed. `app.py` re-reads this folder on every request (cached until
a file changes), so it picks up new/edited skills without a restart.

A soft cap (`_MAX_SKILLS_PER_TURN` in `app.py`, currently 4) keeps a single
request that happens to match several triggers at once from bloating the
prompt — if you add many overlapping skills, prefer distinct trigger phrases
over broad ones.

## Current library

- `3d-website.md` — when to reach for a full 3D/WebGL build, framework choice, performance/SEO rules.
- `threejs.md` — direct Three.js setup/cleanup recipe.
- `gsap.md` — animation timelines and ScrollTrigger.
- `tailwindcss.md` — Tailwind via the Play CDN build.
- `landing-page.md` — high-conversion page structure.
- `animation-on-scroll.md` — IntersectionObserver reveal-on-scroll.
- `css-border-gradient.md`, `css-alpha-masking.md`, `progressive-blur.md` — CSS surface/edge effects.
- `matterjs.md`, `globe-gl.md`, `vantajs.md`, `cobejs.md`, `unicorn-studio.md` — canvas/WebGL libraries for physics, globes, and animated backgrounds.
- `agency-grid-layout-minimal.md`, `premium-website-quality-bar.md` — higher-level visual-direction skills.
- `threeui-reference.md` — how to treat ThreeUI as a design reference given this app's no-build-step constraint.

## Sources

Adapted, condensed, and rewritten from two open-source (MIT-licensed) references
for use in this project's own agent loop:
- https://github.com/MengTo/Skills — the `agent-skills/web-design/*` playbooks.
- https://github.com/MengTo/threeui — the ThreeUI component catalog (design reference only, see `threeui-reference.md`).
