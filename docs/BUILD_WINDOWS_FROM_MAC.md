# Build Windows Executable From a Mac

## Repository root

Run Git and all Cargo commands from:

```bash
cd /Users/sonics/project/peterMath
```

This folder is the real application root because it contains `Cargo.toml`, `src/`,
`assets/`, `Cargo.lock`, `scripts/package_submission.py`, and `.github/workflows/windows-release.yml`.
It also contains `web_html/`, a static fallback that should ship inside the
final artifact.

Do not mix other reference projects or old backup workspaces into this Git root.
The judges and GitHub Actions workflow need the focused Rust app.

## Recommended method: GitHub Actions

This is the safest route because the Windows binary is built on an actual Windows runner.

### Steps

1. Create a new GitHub repository named `peterMath`.
2. In the app root, initialize Git:

```bash
cd /Users/sonics/project/peterMath
git init
git branch -M main
```

3. On your Mac, verify the local app:

```bash
rustup update
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
cargo run --release
```

4. Commit and push:

```bash
git add .
git commit -m "Initialize peterMath native Rust app"
git push
```

5. Open GitHub → Actions → `Build peterMath Windows EXE`.
6. Run the workflow manually or push to `main`.
7. Download the artifact named `peterMath-windows-x64`.
8. Submit that artifact folder to the judges, together with the explanation materials.

If you choose to keep this outer pack under version control as a separate
archive, make it a separate repository. Do not nest one Git repository inside
another for the competition workflow.

## Why GitHub Actions instead of local cross-compilation

macOS-to-Windows Rust cross-compilation is possible, but it can become fragile when GUI/windowing/GPU libraries are involved. A Windows runner avoids most linker and SDK problems.

## Optional local Windows build route

On a Windows machine with Rust installed:

```powershell
cargo build --release
```

The output binary will be:

```text
target\release\peterMath.exe
```

## Optional macOS local preview

On Mac:

```bash
cargo run --release
```

This is for development preview only. It does not replace the Windows artifact.

## Final Windows submission folder

```text
peterMath_windows_submission/
├─ peterMath.exe
├─ START_WINDOWS.bat
├─ 双击运行-评委版.bat
├─ 打开备用网页.bat
├─ 评委入口.html
├─ README_给评委.txt
├─ 作品说明_学生版.md
├─ 参数实验记录表.csv
├─ web_html/
│  ├─ index.html
│  └─ README.md
├─ 3分钟演示视频.mp4
├─ screenshots/
│  ├─ 01_lenia_life_highlight.png
│  ├─ 02_lenia_raw.png
│  └─ 03_judge_mode.png
├─ video/
│  └─ 3分钟演示视频.mp4
└─ data_exports/
   ├─ experiment_001.json
   └─ experiment_001.csv
```

The workflow assembles this folder by running:

```bash
python scripts/package_submission.py --exe target/release/peterMath.exe --out dist/peterMath_windows_submission
```

## Release checklist

- [ ] Git repository root is `peterMath/`.
- [ ] `target/` is ignored by Git.
- [ ] App opens by double-clicking `peterMath.exe`.
- [ ] `START_WINDOWS.bat` starts the native app.
- [ ] `web_html/index.html` opens directly in a browser as fallback.
- [ ] Window title is `peterMath`.
- [ ] App runs without internet.
- [ ] No local server is required.
- [ ] Presets load without external files, or required files are included in the same folder.
- [ ] 任务模式 opens first and automatic explanation remains available.
- [ ] 首屏能看见 `1 选工具 / 2 选任务 / 3 点生命场`。
- [ ] Canvas coachmark and local feedback appear during first interaction.
- [ ] Export works.
- [ ] README tells judges exactly what to click first.
- [ ] A 3-minute video is included as fallback.
