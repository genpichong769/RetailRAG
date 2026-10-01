"""Record non-destructive RetailRAG runtime dependency availability."""

import json
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base import Config


def tcp_probe(host, port):
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return {"status": "PASS", "endpoint": f"{host}:{port}", "error": None}
    except OSError as exc:
        return {"status": "BLOCKED", "endpoint": f"{host}:{port}", "error": str(exc)}


def docker_probe():
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{json .ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        detail = (completed.stdout or completed.stderr).strip()
        return {
            "status": "PASS" if completed.returncode == 0 else "BLOCKED",
            "exit_code": completed.returncode,
            "detail": detail[-1000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "BLOCKED", "exit_code": None, "detail": str(exc)}


def main():
    config = Config()
    result = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "docker": docker_probe(),
        "mysql": tcp_probe(config.MYSQL_HOST, config.MYSQL_PORT),
        "redis": tcp_probe(config.REDIS_HOST, config.REDIS_PORT),
        "milvus": tcp_probe(config.MILVUS_HOST, config.MILVUS_PORT),
        "minio": tcp_probe("localhost", 9000),
        "etcd": tcp_probe("localhost", 2379),
        "fastapi": tcp_probe("localhost", 8003),
        "qwen": {
            "status": "CONFIGURED_UNVERIFIED" if config.DASHSCOPE_API_KEY.strip() else "BLOCKED_API_KEY",
            "credential_source": "host process environment" if config.DASHSCOPE_API_KEY.strip() else None,
            "credential_value_logged": False,
        },
    }
    output = ROOT / "RETAILRAG_RUNTIME_DEPENDENCIES.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
