# Screenshot And Preview Guide

Generated preview images are created with:

```bash
cargo run --bin render_preview
```

They are written to:

```text
peterMath_exports/previews/
```

Use these files for judge-facing screenshots:

- `lenia_hero.png`: main Lenia artwork.
- `judge_mode_reference.png`: Raw Math View, Artistic View, and explanation/evidence panel.
- `reaction_diffusion_texture.png`: secondary reaction-diffusion system.
- `lenia_showcase.png`: duplicate hero image for submission folders that expect showcase naming.
- `reaction_diffusion_showcase.png`: duplicate reaction-diffusion image for submission folders.

Native app exports and evidence packs should be created during final review and placed in the final submission folder, not committed to the repository.
