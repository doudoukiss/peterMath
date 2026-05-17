#!/usr/bin/env python3
"""Build a native-first peterMath judge submission folder.

The package is intentionally simple: it copies the already-built native
executable plus static support files. Judge launchers do not require Python,
Node, Rust, Visual Studio, internet access, or a local server.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "dist" / "peterMath_windows_submission"
DEFAULT_OFFICIAL_NAME = "学校-数学工程创意实践类-小学组-学生姓名"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package peterMath for judges.")
    parser.add_argument(
        "--exe",
        default=None,
        help="Path to the built peterMath executable. Defaults to target/release/peterMath.exe, then target/release/peterMath.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output folder for the judge package.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a zip archive next to the output folder.",
    )
    parser.add_argument(
        "--official-name",
        default=os.environ.get("PETERMATH_OFFICIAL_NAME", DEFAULT_OFFICIAL_NAME),
        help="Folder name used inside the optional zip archive.",
    )
    return parser.parse_args()


def copy_path(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return True


def copy_directory_contents(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        copy_path(item, destination / item.name)
    return True


def locate_executable(value: str | None) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value))
    candidates.extend(
        [
            PROJECT_ROOT / "target" / "release" / "peterMath.exe",
            PROJECT_ROOT / "target" / "release" / "peterMath",
        ]
    )
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        if resolved.exists():
            return resolved
    tried = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find a built peterMath executable. Run `cargo build --release` first.\n"
        f"Tried:\n{tried}"
    )


def write_text(path: Path, content: str, newline: str = "\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.replace("\n", newline), encoding="utf-8")


def windows_launcher() -> str:
    return r"""@echo off
chcp 65001 >nul 2>nul
setlocal
cd /d "%~dp0"

echo peterMath Windows judge launcher
echo Primary work: native Lenia teaching game.
echo No Rust, Node, Python, Visual Studio, internet, or local server is required.
echo.

if exist "%~dp0peterMath.exe" (
  echo Starting peterMath.exe ...
  start "" "%~dp0peterMath.exe"
  echo.
  echo If the native app does not open on this computer, run 打开备用网页.bat.
  timeout /t 5 /nobreak >nul
  exit /b 0
)

echo peterMath.exe was not found. Opening the static web fallback.
call "%~dp0打开备用网页.bat"
"""


def fallback_launcher() -> str:
    return r"""@echo off
chcp 65001 >nul 2>nul
setlocal
cd /d "%~dp0"

if exist "%~dp0web_html\index.html" (
  start "" "%~dp0web_html\index.html"
  exit /b 0
)

echo web_html\index.html was not found.
pause
exit /b 1
"""


def judge_entry_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>peterMath · 评委入口</title>
    <style>
      :root { color-scheme: light; --ink:#172033; --muted:#5c667a; --line:#d9e1ef; --page:#f4f7fb; --card:#fff; --primary:#26316f; --accent:#0f766e; }
      * { box-sizing: border-box; }
      body { margin:0; min-height:100vh; display:grid; place-items:center; background:var(--page); color:var(--ink); font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif; }
      main { width:min(780px, calc(100vw - 32px)); background:var(--card); border:1px solid var(--line); border-radius:8px; padding:28px; box-shadow:0 18px 48px rgba(27,39,72,.12); }
      h1 { margin:0 0 8px; font-size:30px; }
      p, li { color:var(--muted); line-height:1.7; }
      .status { margin:18px 0; padding:12px 14px; border-radius:6px; background:#f8fafc; border:1px solid var(--line); color:var(--ink); }
      .actions { display:flex; flex-wrap:wrap; gap:12px; margin-top:20px; }
      a { display:inline-flex; align-items:center; justify-content:center; min-height:44px; padding:0 18px; border-radius:6px; border:1px solid var(--line); color:var(--primary); text-decoration:none; font-weight:700; }
      a.primary { background:var(--primary); border-color:var(--primary); color:#fff; }
    </style>
  </head>
  <body>
    <main>
      <h1>peterMath：Lenia 数学生命教学游戏</h1>
      <p>推荐评委先双击 <strong>START_WINDOWS.bat</strong> 或直接运行 <strong>peterMath.exe</strong>。打开后按任务卡完成：选工具、选任务、点生命场、看反馈、解锁数学卡片、导出证据。</p>
      <div class="status">本入口页只是说明和分流。正式作品是原生 Windows 程序；网页 fallback 只用于原生程序无法启动的电脑。</div>
      <ol>
        <li>主入口：运行 peterMath.exe，进入任务模式。</li>
        <li>备用入口：打开 web_html/index.html，直接在浏览器中玩简化 Lenia 任务。</li>
        <li>兜底材料：查看 screenshots、video、data_exports 文件夹。</li>
      </ol>
      <div class="actions">
        <a class="primary" href="peterMath.exe">打开 peterMath.exe</a>
        <a href="web_html/index.html">打开网页备用版</a>
        <a href="README_给评委.txt">查看评委说明</a>
      </div>
    </main>
  </body>
</html>
"""


def judge_readme(metadata: dict[str, str]) -> str:
    return f"""peterMath：Lenia 数学生命教学游戏

推荐打开方式：
1. 双击 START_WINDOWS.bat 或 双击运行-评委版.bat。
2. 也可以直接双击 peterMath.exe。
3. 如果 Windows 出现安全提示，请选择“更多信息”->“仍要运行”。这是未签名学生软件常见提示。

备用打开方式：
如果 peterMath.exe 在评审电脑上无法启动，请双击 打开备用网页.bat，或直接打开 web_html/index.html。网页版是离线 Lenia 教学游戏备用窗口；正式作品仍以 peterMath.exe 为主。

建议 3 分钟评审路径：
1. 唤醒生命场：运行并观察连续场自己演化。
2. 塑造生命：选择绘制或盖章，点击/拖动生命场。
3. 半径挑战：只改变一个规则参数并观察指标变化。
4. 证明同一数据：切换数学原始图 / 艺术表达图，并用检查器看同一点。
5. 生成证据报告：导出可复现状态或证据包。
6. 如需完整旁白，再打开“自动讲解”。

作品说明：
peterMath 不是预制动画，也不是只给专业人士看的技术面板。每一帧都由 Lenia 连续场规则实时计算生成。作品重点是把一个简单公式做成可玩的教学任务，让评委先操作，再通过数学卡片、指标和导出证据理解卷积核、增长函数、阻尼、seed 和交互如何产生生命感形态。

提交信息占位：
学校：{metadata["school"]}
组别：{metadata["group"]}
学生：{metadata["student"]}
作品类别：数学工程创意实践类
"""


def final_checklist(metadata: dict[str, str]) -> str:
    return f"""# 提交前最后检查

## 官方提交信息

- 最终文件夹名：`{metadata["official_name"]}`
- 学校：{metadata["school"]}
- 组别：{metadata["group"]}
- 学生：{metadata["student"]}
- 作品名称：peterMath：Lenia 数学生命教学游戏
- 作品类别：数学工程创意实践类

## 评委运行入口

- Windows 评委优先双击 `START_WINDOWS.bat` 或 `双击运行-评委版.bat`。
- 原生程序无法运行时，再双击 `打开备用网页.bat`。
- 不要求评委安装 Rust、Node、Python、Visual Studio、网络服务或本地服务器。

## 玩法检查

- 打开后默认进入任务模式。
- 首屏能看见 `1 选工具 / 2 选任务 / 3 点生命场`。
- 画布有 coachmark 和操作反馈。
- 完成任务后数学卡片解锁。
- 最终任务能导出可复现状态或证据包。

## 材料检查

- `screenshots/` 放关键截图。
- `video/` 放 3 分钟以内玩法视频。
- `data_exports/` 放程序导出的 JSON、PNG 或 evidence pack。
- `web_html/` 保留为备用入口。
"""


def package_manifest(out_dir: Path, metadata: dict[str, str], exe_source: Path) -> dict[str, object]:
    files = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(out_dir).as_posix())
    return {
        "package": "peterMath-windows-x64",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_executable": str(exe_source),
        "official_name": metadata["official_name"],
        "primary_entry": "peterMath.exe",
        "judge_launchers": ["START_WINDOWS.bat", "双击运行-评委版.bat"],
        "fallback_entry": "web_html/index.html",
        "files": files,
    }


def create_zip(out_dir: Path, official_name: str) -> Path:
    zip_path = out_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    root_name = official_name or out_dir.name
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.rglob("*")):
            arcname = Path(root_name) / path.relative_to(out_dir)
            archive.write(path, arcname.as_posix())
    return zip_path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    exe_source = locate_executable(args.exe)
    metadata = {
        "official_name": args.official_name,
        "school": os.environ.get("PETERMATH_SCHOOL", "学校名称待填写"),
        "group": os.environ.get("PETERMATH_GROUP", "小学组/初中组待填写"),
        "student": os.environ.get("PETERMATH_STUDENT", "学生姓名待填写"),
    }

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copy2(exe_source, out_dir / "peterMath.exe")
    copy_path(PROJECT_ROOT / "README.md", out_dir / "README_PROJECT.md")
    copy_path(PROJECT_ROOT / "assets", out_dir / "assets")
    copy_path(PROJECT_ROOT / "web_html", out_dir / "web_html")
    copy_directory_contents(PROJECT_ROOT / "judge_submission_template", out_dir)
    copy_path(PROJECT_ROOT / "screenshots", out_dir / "screenshots")
    copy_path(PROJECT_ROOT / "peterMath_exports" / "previews", out_dir / "previews")

    for directory in ["screenshots", "video", "data_exports", "previews"]:
        (out_dir / directory).mkdir(parents=True, exist_ok=True)

    write_text(out_dir / "START_WINDOWS.bat", windows_launcher(), "\r\n")
    write_text(out_dir / "双击运行-评委版.bat", windows_launcher(), "\r\n")
    write_text(out_dir / "打开备用网页.bat", fallback_launcher(), "\r\n")
    write_text(out_dir / "评委入口.html", judge_entry_html())
    write_text(out_dir / "README_给评委.txt", judge_readme(metadata))
    write_text(out_dir / "提交前最后检查.md", final_checklist(metadata))
    write_text(
        out_dir / "screenshots" / "请把关键截图放在这里.txt",
        "建议截图：任务模式首屏、塑造生命操作反馈、证明同一数据、证据导出、web fallback。\n",
    )
    write_text(
        out_dir / "video" / "请把3分钟以内玩法视频放在这里.txt",
        "建议视频路径：打开程序 -> 选任务 -> 点生命场 -> 看反馈 -> 解锁数学卡片 -> 导出证据。\n",
    )
    write_text(
        out_dir / "data_exports" / "请把证据导出放在这里.txt",
        "建议放入 peterMath 导出的 share-state JSON、snapshot PNG/JSON 或 evidence pack。\n",
    )
    if not (out_dir / "previews").joinpath("README.txt").exists():
        write_text(
            out_dir / "previews" / "README.txt",
            "运行 cargo run --bin render_preview 后，本目录会包含 Lenia 预览图。\n",
        )

    manifest = package_manifest(out_dir, metadata, exe_source)
    write_text(
        out_dir / "PACKAGE_MANIFEST.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

    required = [
        "peterMath.exe",
        "START_WINDOWS.bat",
        "双击运行-评委版.bat",
        "打开备用网页.bat",
        "README_给评委.txt",
        "评委入口.html",
        "web_html/index.html",
        "PACKAGE_MANIFEST.json",
    ]
    missing = [item for item in required if not (out_dir / item).exists()]
    if missing:
        raise FileNotFoundError(f"Package is missing required files: {missing}")

    print(f"Submission package created at {out_dir}")
    if args.zip:
        zip_path = create_zip(out_dir, metadata["official_name"])
        print(f"Zip archive created at {zip_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"package_submission.py: {error}", file=sys.stderr)
        raise SystemExit(1)
