"""실행 없이 UTF-8·파이썬 AST·원본 노트북 구조·모의 전용 경계를 검사합니다."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def validate(root=ROOT):
    started = time.monotonic()
    checked, errors = 0, []
    files = [root / 'engine.py', root / 'tests/test_engine.py',
             root / 'tools/validate_artifacts.py', root / 'requirements.txt',
             root / 'upbit_auto_trader_v5.ipynb']
    source_files = set(files)
    excluded = {'work', 'evidence', '.codex', '.git', '.venv', '__pycache__', 'node_modules', '보존자료'}
    extensions = {'.py', '.json', '.js', '.html', '.css', '.ipynb', '.md', '.txt', '.yml', '.yaml'}
    for folder, directories, names in os.walk(root):
        directories[:] = [d for d in directories if d not in excluded]
        for name in names:
            path = Path(folder) / name
            if path.suffix in extensions or name in {'Dockerfile', '.dockerignore'}:
                source_files.add(path)
    node = shutil.which('node')
    js_checks = 0
    for path in sorted(source_files):
        if path.stat().st_size > 4_000_000:
            errors.append(f'정적 검사 파일 크기 상한 초과: {path.relative_to(root)}')
            continue
        text = path.read_bytes().decode('utf-8', errors='strict')
        if chr(0xfffd) in text or chr(63) * 4 in text:
            errors.append(f'깨진 문자: {path.name}')
        if path.suffix == '.py':
            ast.parse(text, filename=str(path))
        elif path.suffix == '.json':
            json.loads(text)
        elif path.suffix == '.js':
            if node is None:
                errors.append('JavaScript 구문 검사에 필요한 node 실행 파일이 없습니다')
            else:
                remaining = 13 - (time.monotonic() - started)
                if remaining <= 0:
                    errors.append('정적 검사 시간 예산을 초과했습니다')
                    break
                result = subprocess.run([node, '--check', str(path)], shell=False,
                                        capture_output=True, timeout=min(3, remaining), check=False)
                if result.returncode:
                    errors.append('JavaScript 구문 오류: ' + result.stderr.decode('utf-8', errors='strict')[:2000])
                js_checks += 1
        checked += 1
    original = root / '보존자료' / '원본' / 'upbit_auto_trader_v4.ipynb'
    if not original.is_file():
        errors.append('원본 노트북을 찾지 못해 구조 비교가 불가능합니다')
        baseline = None
    else:
        baseline = json.loads(original.read_bytes().decode('utf-8', errors='strict'))
    notebook = json.loads((root / 'upbit_auto_trader_v5.ipynb').read_text(encoding='utf-8'))
    if len(notebook['cells']) != 16:
        errors.append('셀 수가 원본 16개와 다릅니다')
    if baseline is not None:
        if [c['cell_type'] for c in baseline['cells']] != [c['cell_type'] for c in notebook['cells']]:
            errors.append('셀 종류/배치가 다릅니다')
        for i, (old, new) in enumerate(zip(baseline['cells'], notebook['cells'])):
            if old['cell_type'] == 'markdown' and old['source'] != new['source']:
                errors.append(f'원본 마크다운 변경: {i}')
        if baseline['metadata'] != notebook['metadata']:
            errors.append('원본 노트북 메타데이터 변경')
        checked += 3
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            ast.parse(source, filename=f'notebook/cell_{i}')
            if cell.get('outputs') or cell.get('execution_count') is not None:
                errors.append(f'재설계 전 실행 결과가 남아 있습니다: {i}')
            checked += 1
    tree = ast.parse((root / 'engine.py').read_text(encoding='utf-8'))
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and isinstance(n.value, ast.Constant)}
    if constants.get('PAPER_ONLY') is not True:
        errors.append('PAPER_ONLY가 True가 아닙니다')
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {'buy_market_order', 'sell_market_order', 'buy_limit_order', 'sell_limit_order'}:
                errors.append('실제 주문 함수 호출이 있습니다')
        if isinstance(node, ast.Import):
            if any(n.name.split('.')[0] in {'pyupbit', 'telegram', 'ccxt'} for n in node.names):
                errors.append('인증 거래/알림 라이브러리 import가 있습니다')
    source = (root / 'engine.py').read_text(encoding='utf-8')
    if 'method="GET"' not in source or 'CPU 대체 금지' not in source or 'assert_cuda_config' not in source:
        errors.append('읽기 전용 API/GPU 실제 설정 검사 경계가 없습니다')
    checked += 3
    return {'checks': checked, 'errors': errors, 'source_files': len(source_files),
            'javascript_syntax_checks': js_checks, 'duration_seconds': time.monotonic() - started,
            'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)},
            'scope': '소스 UTF-8, Python AST, JSON, JavaScript 구문, 노트북 구조·모의 전용 경계; HTML/CSS는 UTF-8만 검사. 실제 GPU·금융성과·완전 무오류 보증은 아님'}


if __name__ == '__main__':
    result = validate()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(1 if result['errors'] else 0)
