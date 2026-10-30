"""기존 2026 컨테이너의 프로젝트 서비스와 로컬 읽기 전용 게이트웨이를 관리합니다."""

from __future__ import annotations

import argparse
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import threading

if __package__:
    from . import container_runtime as runtime
else:
    import container_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
NAME = "2026"
MAX_BYTES = 8 * 1024 * 1024
COMMAND_TIMEOUT = 20
MOUNT_DESTINATION = "/workspace/2026"


def docker_path() -> str:
    found = shutil.which("docker")
    fallback = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
    if found:
        return found
    if fallback.is_file():
        return str(fallback)
    raise RuntimeError("도커 실행 파일을 찾지 못했습니다.")


def run(args: list[str]) -> str:
    """두 출력 스트림을 합산 제한하며 셸 없이 유한 시간 실행합니다."""
    argv = [docker_path(), *args]
    buffers = [bytearray(), bytearray()]
    total = 0
    lock = threading.Lock()
    overflow = threading.Event()
    failures = []
    with subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, shell=False) as process:
        def drain(stream, target):
            nonlocal total
            try:
                while True:
                    chunk = stream.read1(65536)
                    if not chunk:
                        return
                    with lock:
                        remaining = MAX_BYTES - total
                        buffers[target].extend(chunk[:max(remaining, 0)])
                        total += len(chunk)
                        if total > MAX_BYTES:
                            overflow.set()
                            process.kill()
                            return
            except OSError as error:
                failures.append(error)
            finally:
                stream.close()

        readers = [threading.Thread(target=drain, args=(stream, index), daemon=True)
                   for index, stream in enumerate((process.stdout, process.stderr))]
        for reader in readers:
            reader.start()
        try:
            process.wait(timeout=COMMAND_TIMEOUT)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            raise RuntimeError(f"도커 명령 {COMMAND_TIMEOUT}초 시간 초과: {argv!r}") from error
        finally:
            for reader in readers:
                reader.join(timeout=2)
        if overflow.is_set():
            raise RuntimeError("도커 출력이 8MiB 상한을 초과했습니다.")
        if failures or any(reader.is_alive() for reader in readers):
            raise RuntimeError(f"도커 출력 수집 실패: {failures!r}")
        stdout = buffers[0].decode("utf-8", errors="strict")
        stderr = buffers[1].decode("utf-8", errors="strict")
        if process.returncode:
            raise RuntimeError(f"도커 명령 실패: 종료 {process.returncode}; argv={argv!r}; {stderr or stdout}")
        if stderr:
            print(stderr, file=sys.stderr, end="")
        return stdout


def project_root() -> str:
    return str(PurePosixPath(MOUNT_DESTINATION) / ROOT.name)


def owned_container() -> dict:
    data = json.loads(run(["inspect", NAME]))
    if not isinstance(data, list) or len(data) != 1:
        raise RuntimeError("기존 2026 컨테이너가 없습니다. 생성·변경하지 않았습니다.")
    container = data[0]
    if (container.get("Name") != "/2026" or
            container.get("Config", {}).get("Labels", {}).get("io.yunseok.managed") != "docker2026-v1"):
        raise RuntimeError("2026 관리 라벨이 다릅니다. 다른 프로젝트 컨테이너를 변경하지 않았습니다.")
    mounts = container.get("Mounts", [])
    matching = [mount for mount in mounts if mount.get("Destination") == MOUNT_DESTINATION]
    if (len(matching) != 1 or matching[0].get("Type") != "bind" or matching[0].get("RW") is not True or
            os.path.normcase(str(Path(matching[0].get("Source", "")).resolve())) !=
            os.path.normcase(str(ROOT.parent.resolve()))):
        raise RuntimeError("2026 상위 폴더의 쓰기 가능 마운트 경계가 다릅니다. 실행을 거부합니다.")
    project = PurePosixPath(project_root())
    for mount in mounts:
        destination = PurePosixPath(mount.get("Destination", "/"))
        if mount is not matching[0] and (destination == project or project in destination.parents):
            raise RuntimeError("다른 프로젝트 마운트가 실행 경로를 가립니다. 실행을 거부합니다.")
    if container.get("State", {}).get("Running") is not True:
        raise RuntimeError("2026 컨테이너가 실행 중이 아닙니다. 자동 시작하지 않았습니다.")
    return container


def exec_prefix() -> list[str]:
    return ["exec", "-w", project_root(), NAME, ".venv/bin/python", "-B", "-X", "utf8"]


def execute(action: str) -> str:
    if action not in ("start", "status", "stop", "gpu", "test"):
        raise ValueError("지원하지 않는 프로젝트 명령입니다.")
    owned_container()
    if action in ("start", "status", "stop"):
        command = ["tools/container_runtime.py", action]
    elif action == "gpu":
        command = ["engine.py", "check-gpu"]
    else:
        command = ["-m", "unittest", "discover", "-s", "tests", "-v"]
    return run([*exec_prefix(), *command])


def relay(path: str) -> dict:
    if runtime.path_status(path) != 200:
        raise ValueError("허용되지 않은 중계 경로입니다.")
    owned_container()
    return json.loads(run([*exec_prefix(), "tools/container_runtime.py", "relay", path]))


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "LocalReadOnlyGateway"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(25)

    def reply(self, status: int, body: bytes, headers=None):
        self.send_response(status)
        headers = headers or {"Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store",
                              "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY"}
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        try:
            status = runtime.path_status(self.path)
            if status != 200:
                self.reply(status, b'{"error":"path_not_allowed"}')
                return
            value = relay(self.path)
            status = value["status"]
            if type(status) is not int or not 200 <= status <= 599:
                raise ValueError("잘못된 중계 응답 상태입니다.")
            headers = runtime.filtered_headers(value["headers"].items())
            encoded = value["body_base64"]
            if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_BYTES + 2) // 3):
                raise ValueError("중계 본문이 상한을 초과했습니다.")
            body = base64.b64decode(encoded, validate=True)
            if len(body) > MAX_BYTES:
                raise ValueError("중계 본문이 상한을 초과했습니다.")
            self.reply(status, body, headers)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError, binascii.Error) as error:
            body = json.dumps({"error": "gateway_unavailable", "detail": str(error)}, ensure_ascii=False).encode("utf-8")
            self.reply(503, body)

    def method_not_allowed(self):
        self.reply(405, b'{"error":"read_only"}')

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD = method_not_allowed

    def log_message(self, format, *args):
        # 요청 경로의 임의 입력을 접근 로그에 복제하지 않습니다.
        return


def create_server(port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), GatewayHandler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop", "gpu", "test", "serve"))
    args = parser.parse_args()
    if args.action == "serve":
        with create_server() as server:
            print(execute("start"))
            print("읽기 전용 대시보드: http://127.0.0.1:8765", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    else:
        print(execute(args.action))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
