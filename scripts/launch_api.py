from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Aegis Fraud Detection API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"Starting Aegis API at http://{args.host}:{args.port}")
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print(f"Docs:      http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()