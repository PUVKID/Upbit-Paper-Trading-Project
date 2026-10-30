"""Linux 공유 컨테이너 안에서 현재 프로젝트만 제어하고 HTTP를 중계합니다."""

from __future__ import annotations

import argparse
import ast
import base64
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROC = Path("/proc")
MAX_BYTES = 8 * 1024 * 1024
SAFE_HEADERS = frozenset({"content-type", "cache-control", "content-security-policy",
                          "x-content-type-options", "x-frame-options", "referrer-policy",
                          "permissions-policy"})


def allowed_paths() -> frozenset[str]:
    # 호스트에서도 SQLite 연결을 만들지 않도록 모듈 실행 없이 상수만 읽습니다.
    tree = ast.parse((ROOT / "dashboard.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STATIC" for t in node.targets):
            paths = ast.literal_eval(node.value)
            if not isinstance(paths, dict) or not all(isinstance(p, str) and p.startswith("/") for p in paths):
                break
            return frozenset(paths) | {"/", "/api/status"}
    raise RuntimeError("대시보드 정적 경로 허용 목록을 확인할 수 없습니다.")


def path_status(path: str) -> int:
    if "?" in path or "#" in path:
        return 400
    if "%" in path or "\\" in path or "//" in path or any(ord(c) < 32 for c in path):
        return 404
    if any(part in (".", "..") for part in path.split("/")):
        return 404
    return 200 if path in allowed_paths() else 404


def filtered_headers(headers) -> dict[str, str]:
    result = {}
    for key, value in headers:
        if key.lower() not in SAFE_HEADERS:
            continue
        if not isinstance(value, str) or len(value) > 8192 or any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("중계 응답 헤더가 올바르지 않습니다.")
        value.encode("latin-1", errors="strict")
        result[key] = value
    return result


def relay(path: str) -> dict:
    if path_status(path) != 200:
        raise ValueError("허용되지 않은 중계 경로입니다.")
    connection = http.client.HTTPConnection("127.0.0.1", 8765, timeout=10)
    try:
        connection.request("GET", path, headers={"Accept-Encoding": "identity"})
        response = connection.getresponse()
        headers = filtered_headers(response.getheaders())
        body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise RuntimeError("대시보드 응답 본문이 8MiB 상한을 초과했습니다.")
        return {"status": response.status, "headers": headers,
                "body_base64": base64.b64encode(body).decode("ascii")}
    finally:
        connection.close()


def is_supervisor(pid: int) -> bool:
    try:
        entry = PROC / str(pid)
        if (entry / "cwd").resolve(strict=True) != ROOT:
            return False
        command = (entry / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        expected = [str(ROOT / ".venv/bin/python").encode(), b"-u", str(ROOT / "supervisor.py").encode()]
        return command == expected
    except (FileNotFoundError, ProcessLookupError):
        return False


def processes() -> list[dict]:
    found = []
    for entry in PROC.iterdir():
        if entry.name.isascii() and entry.name.isdecimal() and is_supervisor(int(entry.name)):
            found.append({"pid": int(entry.name), "cwd": str(ROOT), "script": str(ROOT / "supervisor.py")})
    return sorted(found, key=lambda item: item["pid"])


@contextmanager
def project_lock():
    import fcntl

    service = ROOT / "work" / "service"
    service.mkdir(parents=True, exist_ok=True)
    with (service / "runtime.lock").open("a+b") as lock:
        deadline = time.monotonic() + 5
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("프로젝트 제어 잠금 획득 시간 초과")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def start() -> dict:
    current = processes()
    if current:
        return {"state": "running", "processes": current, "started": False}
    console = ROOT / "work/service/supervisor-console.log"
    console.parent.mkdir(parents=True, exist_ok=True)
    with console.open("ab") as stream:
        process = subprocess.Popen([str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "supervisor.py")],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                   shell=False, start_new_session=True, close_fds=True)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"프로젝트 supervisor 시작 실패: 종료 {process.returncode}; {console}")
        if is_supervisor(process.pid):
            return {"state": "running", "processes": [{"pid": process.pid, "cwd": str(ROOT)}], "started": True}
        time.sleep(0.05)
    raise RuntimeError(f"프로젝트 supervisor 신원 확인 시간 초과; {console}")


def stop() -> dict:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("PID 재사용을 방지하는 Linux pidfd 신호 기능이 필요합니다.")
    signalled = []
    for item in processes():
        pid = item["pid"]
        try:
            fd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        try:
            # PID 핸들 고정 뒤 cwd와 전체 argv를 다시 확인하여 다른 프로젝트에 신호를 보내지 않습니다.
            if is_supervisor(pid):
                try:
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
                    signalled.append(pid)
                except ProcessLookupError:
                    continue
        finally:
            os.close(fd)
    return {"state": "stop_requested", "signalled": signalled}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop", "relay"))
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("이 도구는 Linux 컨테이너 내부 전용입니다.")
    if args.action == "relay":
        if args.path is None:
            parser.error("relay에는 허용된 HTTP 경로가 필요합니다.")
        result = relay(args.path)
    else:
        if args.path is not None:
            parser.error("프로세스 제어 명령에는 경로 인자를 사용할 수 없습니다.")
        if args.action == "status":
            current = processes()
            result = {"state": "running" if current else "stopped", "processes": current, "root": str(ROOT)}
        else:
            with project_lock():
                result = start() if args.action == "start" else stop()
    encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise RuntimeError("중계 JSON 출력이 8MiB 상한을 초과했습니다.")
    print(encoded.decode("utf-8"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, http.client.HTTPException) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
