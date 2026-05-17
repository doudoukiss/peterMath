# peterMath Project Brief

本文件记录已经确定的项目方向，并把早期 ChatGPT 方案统一成当前工作区可执行的计划。

## 核心目标

`peterMath` 要参加数学工程创意实践类比赛。它不应被包装成“漂亮动画”或普通网页效果，而应被评审理解为：

```text
Lenia 数学生命教学游戏
```

最终交付给评委的材料应包括：

```text
peterMath.exe + web_html/index.html + 评委说明 + 学生作品说明 + 实验数据 + 截图 + 3分钟演示视频
```

评委打开 `peterMath.exe` 后，不应需要安装 Rust、Node、npm、Visual Studio、Python 或启动本地服务器，也不应先读长篇说明。默认路径是进入任务模式，看见 `1 选工具 / 2 选任务 / 3 点生命场`，完成短任务、看到反馈、解锁数学卡片、导出证据。如果评审电脑无法运行 `.exe`，可以直接打开 `web_html/index.html` 作为备用教学游戏窗口。

## 已确定技术路线

主路线：

```text
Rust + egui/eframe + wgpu/WGSL + GitHub Actions Windows build
```

理由：

1. 可以在 Mac 上用 Codex 和 Rust 工具链开发、预览。
2. 可以让 GitHub Actions 在真实 Windows runner 上构建 `peterMath.exe`。
3. Rust/egui 适合做原生任务面板、指标面板、自动讲解和导出功能。
4. wgpu/WGSL 是后续提升模拟分辨率和视觉质量的正确方向。
5. 这个方向能把作品从“动画展示”升级为“任务、规则、seed、指标、数学卡片、证据导出”组成的教学游戏。

备用路线：

- 静态 Web/HTML：作为提交包内的应急演示窗口，不替代原生主程序。
- Tauri/WASM：只有在原生 Rust/egui 路线遇到严重交付问题时再考虑。
- C++/OpenGL：只有在性能确实成为瓶颈且能承担更复杂构建时再考虑。

## 当前工作区解释

真正应作为 Git 仓库根目录的是：

```text
/Users/sonics/project/peterMath
```

原因：

- 这里有 `Cargo.toml`、`Cargo.lock`、`src/`、`assets/`。
- 这里有 `scripts/package_submission.py` 负责组装评委提交包。
- 这里有 `web_html/` 静态备用演示。
- 这里有 `.github/workflows/windows-release.yml`。
- GitHub Actions workflow 默认从仓库根目录运行 `cargo build --release`。
- 不要把其他参考项目或历史备选路线混进这个 Git 根目录。

## 当前产品状态

`peterMath/` 已经是可继续开发的原生 Rust scaffold：

- 当前提交路线只保留 Lenia 作为主系统。
- 已有 Raw Math View、Artistic View、自动讲解基础。
- 已有基础参数面板、指标显示和 snapshot/parameter 导出。
- 已有 `web_html/index.html` 作为 Lenia 备用网页。
- 已有 GPU Lenia 主路径和 CPU 参考路径。
- 需要把这些能力重组为任务模式，而不是继续扩展成技术面板。

## 后续开发总方向

按优先级推进：

1. 先保证仓库根目录、构建、格式化、测试、打包脚本和 Windows artifact 都可靠。
2. 默认打开后进入任务模式，5 秒内让评委知道 `1 选工具 / 2 选任务 / 3 点生命场`。
3. 把 UI 语言统一为 mission、goal、feedback、math card、field、kernel、seed、metric、evidence。
4. 先不新增其他模拟系统；第一阶段只做 Lenia 教学游戏。
5. 保留自动讲解作为辅助路径，不让它压过任务模式。
6. 做强导出：PNG、参数 JSON、share-state JSON、evidence pack。
7. 最后录制演示视频，并整理最终提交文件夹。

## Codex 使用原则

不要一次性粘贴所有 prompt。一次只做一个阶段：

```text
Prompt 0 -> build and audit
Prompt 1 -> teaching-game UI architecture
Prompt 2 -> mathematical visual language
Prompt 3 -> Lenia
Prompt 5 -> web fallback alignment
Prompt 8 -> mission feedback and metrics
Prompt 9 -> automatic explanation mode
Prompt 10 -> export/submission
Prompt 11 -> Windows artifact
Prompt 12 -> final polish
```

`Prompt 4` 的 GPU/WGSL 迁移应在 CPU 参考模型和 UI 结构稳定后进行。不要为了炫技过早重写全部模拟。

## 最终成功标准

比赛角度的成功不是“代码很多”，而是评委能快速玩明白：

1. 这是可以互动的数学规则系统。
2. 每个任务都有目标、反馈和数学解释。
3. 参数变化会产生可复现结果。
4. 画面美感来自 field/kernel/growth/damping 等规则，而不是装饰动画。
5. 学生能解释核心规则，并用导出的数据支撑观察。
6. Windows 评审机器上可以稳定双击运行。
