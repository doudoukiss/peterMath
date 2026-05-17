# Backup Options

These are fallback and backup routes. The active competition app remains:

```text
peterMath/
```

Do not switch away from the Rust + egui/eframe + wgpu/WGSL route unless the fallback clearly improves competition delivery and presentation quality.

## Static Web/HTML Fallback

This fallback is already part of the app repository:

```text
peterMath/web_html/index.html
```

It is not a replacement for the native app. It is a direct-open emergency teaching-game fallback for judge machines where `peterMath.exe` fails to start. Keep it static: no server, npm, Node, Rust, Python, or internet.

The web fallback should show the same core story: short missions, mathematical rule, deterministic seed, parameter controls, Raw Math View, Artistic View, metrics, and evidence/share actions.

## Tauri/WASM

Use this only if the native Rust/egui route becomes too difficult to package or present.

This route may make sense if:

- a polished web interface already exists and is worth preserving;
- Rust simulation code can be compiled to WASM cleanly;
- Tauri improves Windows delivery without weakening the mathematical visual language.

Risks:

- It remains WebView-based.
- It may keep the project too close to the old HTML-animation direction.
- Windows WebView/runtime behavior adds another compatibility layer.

If explored, keep `peterMath` naming, deterministic seeds, exportable parameter JSON, mission mode, automatic explanation, and a GitHub Actions Windows artifact.

## C++/OpenGL

Use this only if Rust/wgpu cannot reach the required visual performance and the team accepts higher build complexity.

Likely stack:

- CMake
- GLFW or SDL2
- OpenGL 3.3+ or Vulkan
- Dear ImGui
- a simple PNG export library

Risks:

- dependency management;
- DLL bundling;
- Windows build configuration;
- more complex debugging with Codex.

If explored, implement only the current Lenia teaching-game path first, and compare the build risk against the current Rust route before changing direction.
