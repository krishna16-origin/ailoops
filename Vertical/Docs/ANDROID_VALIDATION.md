# Android Build and Validation

## Required free local tooling

This source project needs Unity Hub, Unity **2022.3 LTS**, and the **Android Build Support** module with its bundled Android SDK, NDK, and OpenJDK. The Unity Personal workflow is suitable where the developer is eligible for that free license. The project is designed for Android portrait devices and uses the Android 7.0 minimum API level configured by **Tools → Vertical → Configure Android Player**.

## Build procedure

Open the `Vertical` folder from Unity Hub, allow Unity to resolve packages, and open `Assets/Scenes/Vertical.unity`. Select **Tools → Vertical → Configure Android Player**. Confirm that the target platform is Android in **File → Build Settings**, configure a private signing key under **Project Settings → Player → Publishing Settings**, and select **Build** to produce an APK. Never commit the signing key.

## Functional route to test

| Step | Expected result |
|---|---|
| Launch | The title screen presents Start Ascent, graphics profile, and settings. |
| Settings | Graphics, haptics, audio, and swipe sensitivity all persist when returning to title. |
| Start Ascent → Begin | The tower route starts and the HUD says `TAP CYAN TARGET`. |
| Tap cyan sphere | A cyan rope attaches and the player begins a momentum-preserving swing. |
| Release | The player launches along the swing velocity; directional drag adds a small launch bias. |
| Hold after release | The state changes to glide and descent slows. |
| Double-tap while airborne | The player attempts a forgiving recovery grapple. |
| Land on an amber ledge | A checkpoint toast appears and respawn uses that ledge. |
| Reach the final amber ledge | The player sees the chapter completion story screen. |

## Performance checks

Test first on a physical mid-range Android phone in **Balanced** mode. Use **Performance** when frame pacing is not stable, then profile in Unity's Android Profiler. The starting performance budget is 30 FPS, one shadowed directional light, compact procedural tower geometry, and a capped rain particle system. Test **Cinematic** only on a stronger device.

## Automation status

The source passes repository-integrity and structural C# checks in this workspace. An APK was not generated here because this workspace has no Unity editor or Android build module installed. Actual Unity compilation, Android signing, device installation, and on-device frame profiling remain required before release.
