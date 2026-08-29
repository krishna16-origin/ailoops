---
name: threeui-reference
triggers: threeui, three ui, designcodeio
---
THREEUI REFERENCE SKILL — ThreeUI (`@designcodeio/threeui`, https://threeui.com) is a catalog of prebuilt animated Three.js/WebGL section components (heroes, halftone/shader backgrounds, cursor-trail effects, parallax scenes) distributed as a React npm package that expects a bundler (Vite/Next) and its own runtime asset files.

Code mode here generates plain static HTML/CSS/JS with no package manager or build step, so `npm install @designcodeio/threeui` cannot be dropped into generated output directly. Treat ThreeUI as a design-quality and pattern reference instead: match the *kind* of effect it demonstrates (a restrained shader-driven hero, a halftone cursor trail, a slow parallax scene) using the vanilla Three.js setup in the threejs/3d-website skills, GSAP for choreography, and plain CSS for the surrounding layout — never claim to have installed or imported the actual package.

If the user specifically wants the real ThreeUI React components (not just the visual style), that requires a different project type than code mode currently builds: a React + Vite (or Next.js) project with a `package.json` listing `@designcodeio/threeui`, an actual `npm install`/build step, and the component's runtime asset files copied into the public directory per ThreeUI's own README. Flag this explicitly rather than silently generating non-functional import statements into a static HTML file.
