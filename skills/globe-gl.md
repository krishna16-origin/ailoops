---
name: globe-gl
triggers: globe.gl, spinning globe, 3d globe, interactive globe with data
---
GLOBE.GL SKILL — a data-driven 3D globe (points/arcs/polygons/labels) built on Three.js under the hood. Load via CDN script tag:

```html
<div id="globe" style="width: 100%; height: 500px;"></div>
<script src="https://cdn.jsdelivr.net/npm/globe.gl"></script>
<script>
  const myGlobe = new Globe(document.getElementById("globe"))
    .globeImageUrl("//unpkg.com/three-globe/example/img/earth-dark.jpg")
    .pointsData(myData); // array of { lat, lng, size, color, ... }
</script>
```

Available layers to reach for depending on the request: points, arcs, polygons, paths, heatmaps/hex bins, labels or HTML elements, custom 3D objects. Size the container with CSS — the globe fills its parent. Reduce point count/size on mobile for performance, and use a darker globe texture for neon-style data overlays.
