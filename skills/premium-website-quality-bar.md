---
name: premium-website-quality-bar
triggers: awwwards, awwwards-quality, premium website, cinematic website, high-concept website, motion-led website, agency-quality site
---
PREMIUM WEBSITE QUALITY-BAR SKILL — apply when the request explicitly asks for an Awwwards-quality, premium, cinematic, or motion-led site. Build a cohesive site whose visual idea, imagery, typography, and motion all tell the same story. Treat "Awwwards quality" as an acceptance bar, never as an award or recognition claim — never describe the result as award-winning.

1. Art direction: generate a genuinely new visual identity, layout, copy, and interaction language for this request — never trace or closely reproduce a specific site a user references. Write a one-paragraph direction before coding: visual thesis, hero focal idea, type hierarchy, color system, section sequence, motion narrative, and whether Three.js is justified.

2. Honest assets: never fabricate fake photos of real people as testimonials/avatars; use plain initials or simple illustrated avatars instead when no real photo is supplied. Use simple authored icons/SVG marks for interface symbols. Skip a logo wall if there's no real, honest proof to show.

3. Hero: make the first viewport the strongest authored moment — clear message + CTA, combined with an original visual (imagery, a justified Three.js scene, or a composed GSAP intro). Keep nav, message, and CTA usable before any animation finishes, and design a complete static first frame for when JS/motion is unavailable.

4. Motion system: GSAP as the primary animation system (see the gsap skill); pick section-by-section choreography over scattered one-off effects; reveal headings with a restrained word/line stagger; use plain CSS for simple hover/focus states and reserve ScrollTrigger for scroll-driven sequences; always bypass scrub/pin timelines under `prefers-reduced-motion: reduce` and show the end state immediately instead.

5. Three.js with purpose only (see the 3d-website skill for the framework/performance rules) — add it when spatial depth or pointer-responsive shading genuinely supports the concept, never as ornamental background noise, and give the canvas one clear job.

6. Quality bar: a complete page, not a hero-only concept — real nav, section progression, concrete content, footer, working form/control states, visible keyboard focus. Reject generic gradient blobs, glass-everywhere, stock bento grids, and motion with no narrative role.

7. Before finishing: verify keyboard navigation and visible focus, a usable no-JS fallback, and `prefers-reduced-motion` behavior; make sure any WebGL/animation resources get cleaned up (see the 3d-website and gsap skills); double-check there are no placeholder claims, invented logos, or unsupported "award-winning" language in the copy.
