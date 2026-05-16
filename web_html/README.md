# peterMath Web Fallback

This folder contains a standalone HTML fallback for judges to open if the Windows executable cannot run on a specific computer.

Open:

```text
index.html
```

No server, Node, npm, Python, Rust, internet connection, or browser extension is required.

## Purpose

The native `peterMath.exe` remains the primary submission. This web fallback is a backup demonstration window that preserves the same judging story:

- mathematical rules generate the visual field;
- parameters change the pattern;
- deterministic seeds make experiments repeatable;
- Raw Math View and Artistic View show the same underlying data differently;
- metrics summarize the evolving field.

## Sharing

The fallback supports URL hash sharing. A link can preserve preset, seed, parameters, view style, and speed:

```text
index.html#preset=labyrinth&seed=4101&feed=0.036&kill=0.060&diffA=1.00&diffB=0.50&style=artistic&steps=4
```

Use `Copy share link` inside the page to generate the current link. Use `Export snapshot` to download the visible canvas as a PNG.

The fallback demonstrates a Gray-Scott reaction-diffusion field because it is compact, reliable in a browser, and visually clear for judges. It is not a replacement for the native GPU Lenia artwork.
