# peterMath Mac→Windows 原生升级工作区

本工作区的主线不是继续优化 HTML 版本，而是为 **Mac 上使用 Codex 开发、最终向评委交付 Windows 可运行程序** 准备的工程起点。HTML 只保留为备用演示窗口，防止评审电脑无法运行 `.exe`。

当前主目标：

```text
peterMath：Lenia 数学生命教学游戏
```

比赛成功标准不是“画面好看”本身，而是评委能快速玩明白：作品由数学规则、任务、参数、seed、指标和证据导出驱动，最终能在 Windows 上双击运行。

## 核心目标

- 开发端：Mac + Codex + Rust 工具链。
- 产品端：Lenia teaching game，突出 mission、goal、feedback、math card、field、kernel、seed、metric、evidence。
- 视觉端：优先采用原生 Rust + `wgpu`/`egui`，以连续场、细密纹理、参数系统为主，不再依赖大颗粒动画。
- 交付端：通过 GitHub Actions 的 Windows runner 生成 `peterMath.exe` 或 Windows 压缩包，评委无需安装 Rust、Node、Visual Studio 或其他开发依赖。
- 应急端：随提交包附带 `web_html/index.html`，在 `.exe` 无法启动时可直接用浏览器打开。
- 备用路线：保留 Tauri/WASM 与 C++/OpenGL 方案说明，但不作为第一选择。

## 为什么首选 Rust + wgpu + egui

1. **Mac 开发自然**：Rust、egui、wgpu 都适合 macOS 开发。
2. **Windows 交付清晰**：可以让 GitHub Actions 在 Windows 环境构建，输出 `.exe`。
3. **视觉上更接近数学系统**：当前只用 Lenia 连续场和 GPU shader 表达，避免回到网页粒子动画。
4. **可解释性更强**：UI 可固定为“系统模式 + 参数 + 公式说明 + 实验指标 + 导出”。

## 本工作区结构

```text
peterMath/
├─ README.md
├─ docs/
│  ├─ PROJECT_BRIEF.md                # 项目方向、路线、阶段目标
│  ├─ BUILD_WINDOWS_FROM_MAC.md       # Mac 开发、Windows 构建流程
│  ├─ CODEX_PROMPTS_EN.md             # 分阶段 Codex prompts
│  ├─ REFERENCE_PROJECTS.md           # 可研究的外部项目
│  └─ BACKUP_OPTIONS.md               # Tauri/WASM 与 C++/OpenGL 备用路线
├─ scripts/package_submission.py      # 评委提交包组装脚本
├─ web_html/                          # 静态 HTML 备用演示窗口
└─ judge_submission_template/         # 评委提交模板
```

## 正确的 Git 根目录

应在下面这个目录运行 `git init`：

```bash
cd /Users/sonics/project/peterMath
git init
git branch -M main
```

正式比赛仓库应聚焦在这个 Rust 应用根目录本身，这样 GitHub Actions、README、源码、assets、打包脚本和提交模板都在同一个清晰根目录下。

## 建议工作流

1. 将 `peterMath/` 作为新的 GitHub 仓库根目录。
2. 在 Mac 上安装 Rust 后运行：

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
cargo run --release
cargo run --bin render_preview
python3 scripts/package_submission.py --out dist/peterMath_windows_submission
```

正式提交前可补充元数据：

```bash
python3 scripts/package_submission.py --out dist/peterMath_windows_submission --zip --official-name "学校-数学工程创意实践类-小学组-学生姓名" --school "学校名称" --group "小学组" --student "学生姓名"
```

3. 把 `docs/CODEX_PROMPTS_EN.md` 中的 Prompt 0 到 Prompt 12 按顺序交给 Codex。
4. 每完成一个阶段都运行：

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test
cargo run --release
```

5. 推送到 GitHub 后，在 Actions 中运行 `Build peterMath Windows EXE` workflow。
6. 下载 GitHub Actions artifact，最终给评委的是：

```text
peterMath_windows_submission/
├─ peterMath.exe
├─ START_WINDOWS.bat
├─ 双击运行-评委版.bat
├─ 打开备用网页.bat
├─ 评委入口.html
├─ README_给评委.txt
├─ 作品说明_学生版.docx 或 .md
├─ 参数实验记录表.csv
├─ web_html/index.html
├─ 3分钟演示视频.mp4
└─ 原创与AI规范使用声明.docx
```

## 当前骨架的性质

`peterMath/` 是一个 **Codex-ready teaching game scaffold**：它已有 Lenia 模拟、UI、参数、指标、导出与 GPU/CPU 路径。当前重点不是继续添加系统，而是把评审入口改成短任务、反馈和数学卡片。

当前提交路线只保留 Lenia 作为主系统。不要在第一阶段重新扩展其他模拟系统，否则会削弱“评委打开即可玩”的核心体验。

同时保留 `peterMath/web_html/index.html` 作为静态网页备用窗口。它不是主作品路线，而是评审电脑无法运行 `peterMath.exe` 时的应急教学游戏。

评委提交模板现在放在正式 app 根目录下：

```text
peterMath/judge_submission_template/
```

## 下一步优先级

1. 先修好基础工程卫生：格式化、测试、`.gitignore`、GitHub Actions artifact 和打包脚本。
2. 强化默认任务模式，让打开后 5 秒内知道 `1 选工具 / 2 选任务 / 3 点生命场`，并能直接点击推荐工具。
3. 让五个任务覆盖运行、塑形、调参、同一数据证明和证据导出。
4. 加强导出：PNG、参数 JSON、share-state JSON、evidence pack。
5. 确认 GitHub Actions artifact 同时包含 `peterMath.exe`、Windows 启动脚本和 `web_html/`。
6. 自动讲解和专家面板作为辅助，不压过任务模式。

重点不是把所有算法一次写完，而是把工程骨架、任务体验、数学解释和 Windows 交付链打通。
