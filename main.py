#!/usr/bin/env python3
"""NutriLife 后端启动入口。

用法::

    python main.py                 # 默认 0.0.0.0:8000
    python main.py --reload        # 开发模式（热重载）
    python main.py --port 9000     # 指定端口
    python main.py --host 127.0.0.1

也可通过环境变量覆盖：HOST / PORT / RELOAD。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 切换到项目根目录：保证 .env、data/ 等相对路径正确，
# 并确保 reload 子进程能通过 cwd 导入 app 包。
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _env_flag(name: str, default: bool = False) -> bool:
    """读取布尔型环境变量（1/true/yes/on 视为 True）。"""
    return (
        os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
        if os.getenv(name)
        else default
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NutriLife 后端启动入口")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="监听地址")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("PORT", "8000")), help="监听端口"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=_env_flag("RELOAD"),
        help="开发模式热重载",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("未安装 uvicorn，请先安装依赖：", file=sys.stderr)
        print(
            "  pip install -r requirements.txt   # 或  poetry install", file=sys.stderr
        )
        sys.exit(1)

    print(
        f"NutriLife 后端启动中：http://{args.host}:{args.port}  (reload={args.reload})"
    )
    print(f"API 文档：http://localhost:{args.port}/docs")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
