# Vertical

**Vertical** is a playable 3D mobile traversal game built entirely with the free and open-source Godot Engine. The Android game is portrait-first and uses a one-thumb grammar: tap cyan grappling nodes, release to launch, and hold during a fall to glide.

The project uses the GLES3-compatible mobile renderer and targets `armeabi-v7a` and `arm64-v8a`. The tower, targets, lighting, effects, HUD, and menu flows are generated at runtime from GDScript, so no proprietary game engine or licensed editor is required.

## Android build

The bundled `export_presets.cfg` contains a debug Android preset. In this workspace, build with the command below.

```bash
export ANDROID_SDK_ROOT=/home/ubuntu/Android/Sdk
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
/home/ubuntu/tools/godot/Godot_v4.7.1-stable_linux.x86_64 --headless --path /home/ubuntu/ailoops/Vertical --export-debug Android build/Vertical-debug.apk
```

The debug APK is suitable for direct device testing. A separate private signing key is only necessary for a release APK or app-store distribution.
