from src.agent.api import app
from pathlib import Path
import os
import socket


def find_available_port(start_port: int = 8510, max_tries: int = 20) -> int:
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"无法在 {start_port}-{start_port + max_tries - 1} 范围内找到可用端口")


def save_active_port(port: int) -> None:
    project_root = Path(__file__).resolve().parent
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "agent_api_port.txt").write_text(str(port), encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    preferred_port = int(os.getenv("AGENT_API_PORT", "8510"))
    port = find_available_port(start_port=preferred_port, max_tries=30)
    save_active_port(port)
    print(f"[Agent API] 启动端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
