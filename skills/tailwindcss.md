---
name: tailwindcss
triggers: tailwind, tailwindcss, utility classes
---
TAILWIND CSS SKILL — code mode builds plain static HTML/CSS/JS with no build step, so use the Tailwind Play CDN build instead of a compiled pipeline: `<script src="https://cdn.tailwindcss.com"></script>` in `<head>`. Every utility class is available at runtime with this build, so — unlike a compiled Tailwind setup — dynamically constructed class strings (e.g. `"text-" + color`) are not a purge risk here, though static class names are still clearer to read.

Patterns: compose utilities directly in HTML (`class="flex gap-4 p-6 bg-zinc-950 text-white"`); responsive variants (`sm: md: lg: xl:`); state variants (`hover:`, `focus:`, `active:`, `group-hover:`); dark mode via the `dark:` class strategy; use arbitrary values sparingly (`w-[42rem]`, `bg-[#0b1220]`).

Recipes:
```html
<!-- CTA button -->
<button class="inline-flex items-center justify-center rounded-xl px-5 py-3
               bg-indigo-600 text-white font-medium hover:bg-indigo-500
               active:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-400/60">
  Get started
</button>

<!-- Responsive hero -->
<section class="mx-auto max-w-6xl px-6 py-16">
  <div class="grid gap-10 lg:grid-cols-2 lg:items-center">
    <div>
      <h1 class="text-4xl font-semibold tracking-tight sm:text-5xl">Ship a beautiful site fast.</h1>
      <p class="mt-4 text-zinc-600">Tailwind helps you move quickly without fighting CSS.</p>
    </div>
    <div class="rounded-2xl border border-zinc-200 bg-white p-6 shadow-sm"><!-- media --></div>
  </div>
</section>
```

Avoid huge, unstructured class lists on one element — break markup into logical sections instead. Follow the visual-design rules already in force (rich confident colors, no exhausted templates) rather than defaulting to generic Tailwind starter-kit look.
