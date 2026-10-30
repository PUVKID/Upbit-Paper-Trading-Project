"""공유 컨테이너를 건드리지 않고3차 수집기만 제어합니다."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SCRIPT = ROOT / 'external_data.py'


def matches(pid):
    try:
        proc = Path('/proc') / str(pid)
        if proc.joinpath('cwd').resolve() != PROJECT.resolve():
            return False
        arguments = proc.joinpath('cmdline').read_bytes().split(b'\0')
        values = [a.decode('utf-8') for a in arguments if a]
        scripts = [a for a in values[1:] if not a.startswith('-')]
        return (len(scripts) == 2 and (PROJECT / scripts[0]).resolve() == SCRIPT.resolve()
                and scripts[1] == 'run')
    except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeError):
        return False


def processes():
    return [int(p.name) for p in Path('/proc').iterdir() if p.name.isdecimal() and matches(int(p.name))]


def stop():
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('PID 재사용 방지 기능이 없는 환경에서는 정지하지 않습니다')
    found = processes()
    for pid in found:
        fd = os.pidfd_open(pid)
        try:
            if matches(pid):
                signal.pidfd_send_signal(fd, signal.SIGTERM)
        finally:
            os.close(fd)
    deadline = time.monotonic() + 100
    while any(matches(pid) for pid in found):
        if time.monotonic() >= deadline:
            raise TimeoutError('수집기 정상 종료 시간 초과; 강제 종료하지 않았습니다')
        time.sleep(.25)
    return found


def start():
    current = processes()
    if current:
        return current
    (ROOT / 'work').mkdir(parents=True, exist_ok=True)
    with (ROOT / 'work/console.log').open('ab') as stream:
        child = subprocess.Popen([str(PROJECT / '.venv/bin/python'), '-u', str(SCRIPT), 'run'],
                                 cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError(f'수집기 시작 실패: 종료코드{child.returncode}')
        if matches(child.pid):
            return [child.pid]
        time.sleep(.1)
    raise RuntimeError('수집기 신원 확인 시간 초과')


def main():
    if sys.platform != 'linux':
        raise RuntimeError('기존2026 Linux 컨테이너 전용 도구입니다')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'])
    args = parser.parse_args()
    if args.action in ('stop', 'restart'):
        stop()
    pids = start() if args.action in ('start', 'restart') else processes()
    print(json.dumps({'action': args.action, 'pids': pids, 'shared_container_restarted': False}))


if __name__ == '__main__':
    main()
