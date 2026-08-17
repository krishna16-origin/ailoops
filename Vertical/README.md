# Vertical — Unity Android Vertical Slice

This repository contains a Unity 2022.3 LTS project targeting Android portrait devices. It implements the playable core of **Vertical**: a one-thumb third-person grapple, swing, launch, glide, checkpoint, and objective ascent through a storm-noir tower facade.

## Open and run

Install the free **Unity Hub**, Unity **2022.3 LTS**, and the **Android Build Support** module with SDK, NDK, and OpenJDK. In Unity Hub, select **Open** and choose this `Vertical` folder. Unity resolves the URP dependency on first open. Open `Assets/Scenes/Vertical.unity` and press Play.

The game creates its playable scene at runtime, so its tower, player, targets, UI, lighting, rain, and camera do not require drag-and-drop setup. The first screen has a functional title, briefing, quality selector, and start action. Gameplay uses touch input on Android and mouse input in the Unity editor.

## Android build

Switch platform to Android in **File → Build Settings**, set a release signing key in **Project Settings → Player → Publishing Settings**, then use **Build** to create an APK. The configuration targets Android 7.0+ and ARMv7 plus ARM64. A Unity editor with Android support is required to produce the APK; the current automation workspace contains the project source but does not have a Unity editor installed.

## Controls

| Input | Gameplay action |
|---|---|
| Tap near a cyan target | Grapple to a forgiving in-range anchor. |
| Release while attached | Launch along swing velocity. |
| Hold while airborne | Glide while holding the gesture surface. |
| Double-tap | Detach from the rope, or attempt an emergency recovery. |
| Keyboard in editor | `A`/`D` steer; `Space` taps the centre; `R` restarts. |

## Mobile graphics

The default is **Balanced**. Cinematic quality is intended for stronger devices; Performance reduces post-effects and rain. The project favors a high-end noir art direction with mobile-safe URP choices: a single shadowed directional light, static geometry, emissive neon, depth fog, local particles, and conservative dynamic lights.
