---
name: threejs
triggers: three.js scene, threejs scene, spinning object, product viewer, 3d model viewer, orbitcontrols, gltf, glb model
---
THREE.JS RECIPE SKILL — practical setup/cleanup patterns for a real Three.js scene, on top of the 3D-website skill's design rules.

Core mental model: Scene (root graph) + Camera (Perspective/Orthographic) + Renderer (WebGLRenderer) + Mesh (Geometry + Material) + Lights, updated inside a requestAnimationFrame loop.

Minimal setup (CDN ES module import, no bundler):
```html
<script type="module">
import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";

const canvas = document.querySelector("#c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 0, 4);

const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0x7c3aed }));
scene.add(mesh);
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(2, 2, 2);
scene.add(dir);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function animate(t) {
  mesh.rotation.y = t * 0.0006;
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
</script>
```

Loading assets: GLTFLoader for models (only if the user supplies/links one — see the asset rules in the 3D-website skill), TextureLoader for images. Controls: OrbitControls for a product-viewer feel, custom pointer handlers for a hero scene.

Cleanup (required before replacing/removing a scene): `geometry.dispose()`, `material.dispose()`, `texture.dispose()`, `renderer.dispose()`, remove event listeners, cancel the RAF loop.

Pitfalls: no resize handler → stretched/cropped render; uncapped devicePixelRatio → mobile GPU meltdown; not disposing geometries/materials/textures → leaks on rebuild; huge textures/models with no compression pipeline available here → keep them modest by construction.
