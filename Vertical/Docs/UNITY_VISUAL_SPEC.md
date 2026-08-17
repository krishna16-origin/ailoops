# Vertical — Unity Mobile Visual Specification

## Visual target

The Unity vertical slice uses the **Universal Render Pipeline (URP)** in portrait orientation. Its visual goal is not desktop-style indiscriminate effects; it is a cinematic, readable noir ascent whose best details are visible on a mid-range Android phone. Wet concrete, cyan navigation light, selective amber utility lighting, animated rain, and depth fog create the storm-soaked tower without a fully dynamic city or expensive global illumination.

| Area | High-quality mobile implementation | Constraint |
|---|---|---|
| Tower facade | Modular low-poly meshes, trim-sheet materials, vertex variation, static batching | No unique room interiors or high-density mesh detail. |
| Lighting | One realtime directional moon key light, baked static lighting, reflection probe at the player tier, emissive neon | No realtime global illumination, realtime shadows only for the player. |
| Rain | GPU-friendly particle system with a small local splash layer | Hard cap on active particles; particles fade before the far fog layer. |
| Fog | Height-based fog via URP Shader Graph and alpha cards | Reduced in the low graphics tier. |
| Bloom | URP post-processing bloom on cyan and amber emissives | HDR and bloom disabled in performance mode. |
| Character | Stylised silhouette with emissive harness accent and simple two-bone cape motion | Character mesh and material count remain compact. |
| Camera | Third-person shoulder camera with Cinemachine-style smoothing in a vertical 9:16 composition | Field of view opens only during glide. |

## Quality tiers

| Tier | Target device class | Render scale | Shadows | Rain | Fog and post-processing |
|---|---|---:|---|---|---|
| Performance | 2022 mid-range Android baseline | 0.72 | Player-only, 1024 map | Sparse | Height fog, no bloom |
| Balanced | Typical modern Android | 0.85 | Player-only, soft 1536 map | Medium | Height fog and restrained bloom |
| Cinematic | Recent high-end Android | 1.00 | Player and near facade, soft 2048 map | Dense local rain | Height fog, bloom, vignette and color grading |

The game starts in **Balanced**, lets players choose another tier in Settings, and automatically falls back if the moving average frame time exceeds the 30 FPS budget.

## One-thumb interaction model

The lower two-thirds of the screen is a gesture zone. A tap casts a ray to a target point; the game scores all nearby grapple anchors and uses the highest visible candidate rather than requiring a pixel-perfect touch. A sustained hold after a release triggers the glide state. A short directional drag stores a launch bias. A double-tap detaches or triggers a recovery grapple if a target is eligible. The controls never use a permanent virtual joystick.

## Vertical-slice content

The first playable build contains **Chapter 1: The Service Spine**. The player leaves a maintenance terrace, learns two chained grapples, uses a glide gap, passes a broken signage field, reaches the first secure ledge, and receives the first compact story reveal. The entire route is a discrete 3D scene built from repeated facade modules and pooled targets.

## Acceptance checks

The mobile vertical slice is ready for Android packaging when a player can begin the mission, reach the first checkpoint, chain grapples, gain and retain swing momentum, launch, glide, recover or respawn after a miss, finish the objective, and return to the title screen without a dead-end interaction. The game should maintain clear target feedback during rain and fog at all quality settings.
