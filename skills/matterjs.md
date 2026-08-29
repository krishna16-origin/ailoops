---
name: matterjs
triggers: matter.js, matterjs, 2d physics, physics simulation, falling objects physics
---
MATTER.JS SKILL — 2D physics via CDN (`https://cdn.jsdelivr.net/npm/matter-js@0.19.0/build/matter.min.js`).

Minimal setup:
```html
<script>
  const { Engine, Render, Runner, Bodies, Composite } = Matter;
  const engine = Engine.create();
  const render = Render.create({
    element: document.getElementById("physics-container"),
    engine, options: { width: 800, height: 600, wireframes: false },
  });
  const runner = Runner.create();
  Runner.run(runner, engine);
  Render.run(render);

  const ground = Bodies.rectangle(400, 610, 810, 60, { isStatic: true });
  const box = Bodies.rectangle(400, 200, 80, 80);
  Composite.add(engine.world, [ground, box]);
</script>
```

Mouse drag interaction (optional):
```js
const { Mouse, MouseConstraint } = Matter;
const mouse = Mouse.create(render.canvas);
const mouseConstraint = MouseConstraint.create(engine, { mouse });
Composite.add(engine.world, mouseConstraint);
render.mouse = mouse;
```

Set `wireframes: false` for solid rendering (the default wireframe look is for debugging only). Cleanup before removing a scene: `Runner.stop(runner)`, remove the render canvas from the DOM.
