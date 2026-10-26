"""자동매매 이전본을 내용 주소 방식으로 보존하고 삭제 전 해시를 검증합니다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (Path('D:/2026/비트코인 자동 매매'), Path('D:/2026/2026 도커/자동매매'))


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    if ROOT != Path('D:/2026/자동매매') or not all(p.is_dir() for p in SOURCES):
        raise ValueError('통합 대상 경계가 일치하지 않습니다')
    archive = ROOT / '보존자료'
    if archive.exists():
        raise ValueError('이미 생성된 보존 자료를 덮어쓰지 않습니다')
    archive.mkdir()
    known = set()
    entries = []
    bundle = archive / '이전_고유내용.zip'
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for source in SOURCES:
            for path in source.rglob('*'):
                if path.is_symlink():
                    raise ValueError(f'링크는 자동 처리하지 않습니다: {path}')
                if not path.is_file():
                    continue
                sha = digest(path)
                member = 'objects/' + sha
                if sha not in known:
                    output.write(path, member)
                    known.add(sha)
                entries.append({'source': str(path), 'sha256': sha, 'bytes': path.stat().st_size,
                                'stored_as': member})
    with zipfile.ZipFile(bundle) as saved:
        for member in saved.namelist():
            value = hashlib.sha256()
            with saved.open(member) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    value.update(chunk)
            if value.hexdigest() != member.removeprefix('objects/'):
                raise ValueError(f'보존 묶음 해시 불일치: {member}')
    # 원본 노트북은 구조 비교를 위해 사용 가능한 경로에 한 번만 보관합니다.
    original = SOURCES[0] / 'upbit_auto_trader_v4.ipynb'
    notebook = archive / '원본' / original.name
    notebook.parent.mkdir(exist_ok=True)
    shutil.copy2(original, notebook)
    report = {'roots': list(map(str, SOURCES)), 'canonical_root': str(ROOT),
              'files_verified': len(entries), 'entries': entries, 'deletion_performed': False}
    (archive / '이전_대응표.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'verified_files': len(entries), 'unique_objects': len(known),
                      'manifest': str(archive/'이전_대응표.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
