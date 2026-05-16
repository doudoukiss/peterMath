# peterMath Web Fallback

This folder contains a standalone HTML fallback for judges to open if the Windows executable cannot run on a specific computer.

Open:

```text
index.html
```

No server, Node, npm, Python, Rust, or internet connection is required.

## Purpose

The native `peterMath.exe` remains the primary submission. This web fallback is a backup demonstration window that preserves the same judging story:

- mathematical rules generate the visual field;
- parameters change the pattern;
- deterministic seeds make experiments repeatable;
- Raw Math View and Artistic View show the same underlying data differently.

The fallback currently demonstrates a Gray-Scott reaction-diffusion field because it is compact, reliable in a browser, and visually clear for judges.
