param([switch]$ReadyToMove)
$ErrorActionPreference = 'Stop'
if (-not $ReadyToMove) { throw '모든 구현 소스 동결 확인이 필요합니다.' }
$source = [IO.Path]::GetFullPath('D:\2026\자동매매')
$target = [IO.Path]::GetFullPath('D:\2026\Upbit 5-Coin Project')
$parent = [IO.Path]::GetFullPath('D:\2026').TrimEnd('\')
if ([IO.Path]::GetDirectoryName($source) -ne $parent -or [IO.Path]::GetDirectoryName($target) -ne $parent) {
    throw '프로젝트 이름 변경 대상이 승인된 D:\2026 경계를 벗어납니다.'
}
if (-not (Test-Path -LiteralPath $source -PathType Container) -or (Test-Path -LiteralPath $target)) {
    throw '이전 원본 또는 새 경로 상태가 예상과 다릅니다.'
}
$docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
$old = (& $docker inspect upbit-paper-research-v7 | ConvertFrom-Json)[0]
if ($LASTEXITCODE -ne 0 -or $old.Config.Labels.'kr.yunseok.upbit.root' -ne $source) {
    throw '삭제 대상 전용 컨테이너의 프로젝트 표식이 다릅니다.'
}
$shared = (& $docker inspect 2026 | ConvertFrom-Json)[0]
if ($LASTEXITCODE -ne 0 -or -not $shared.State.Running -or $shared.Config.Labels.'io.yunseok.managed' -ne 'docker2026-v1') {
    throw '기존2026 컨테이너를 확인하지 못했습니다.'
}
$before = Get-Content -LiteralPath (Join-Path $source 'work\paper\state.json') -Raw | ConvertFrom-Json
$report = [ordered]@{ started_kst=(Get-Date -Format o); source=$source; target=$target;
    old_container_id=$old.Id; old_image=$old.Image; shared_id=$shared.Id; shared_started=$shared.State.StartedAt;
    initial_cash=$before.initial_cash; cash_before=$before.cash; equity_before=$before.equity;
    account_start=$before.start_time; host_sqlite=$false; other_container_stopped=$false }
try {
    & $docker stop --time 60 $old.Id
    if ($LASTEXITCODE -ne 0) { throw '전용 자동매매 컨테이너 정상 종료 실패' }
    $verify = @'
import sqlite3,json,hashlib
from pathlib import Path
p=Path('/workspace/2026/자동매매/work/data/candles.sqlite3')
c=sqlite3.connect(p,timeout=120)
try:
    checkpoint=c.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchall()
    quick=c.execute('PRAGMA quick_check(1)').fetchall()
    if quick != [('ok',)]: raise RuntimeError(str(quick))
finally: c.close()
h=hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
print(json.dumps({'checkpoint':checkpoint,'quick_check':'ok','sha256':h.hexdigest(),'bytes':p.stat().st_size}))
'@
    $result = & $docker exec 2026 python -c $verify
    if ($LASTEXITCODE -ne 0) { throw 'Linux 무결성 검사 실패. 이름 변경을 중단합니다.' }
    $report.database_before_move = $result | ConvertFrom-Json
    $previews = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match 'dashboard\.py --host 127\.0\.0\.1 --port 8766'
    }
    foreach ($process in $previews) { Stop-Process -Id $process.ProcessId }
    $report.preview_stopped = @($previews.ProcessId)
    # 경계를 위에서 확인한 같은 부모 경로 안에서 폴더 이름만 변경합니다.
    Move-Item -LiteralPath $source -Destination $target
    if ((Test-Path -LiteralPath $source) -or -not (Test-Path -LiteralPath (Join-Path $target '.codex\routing-project.json'))) {
        throw '폴더 이름 변경 결과가 예상과 다릅니다.'
    }
    $sharedAfter = (& $docker inspect 2026 | ConvertFrom-Json)[0]
    if ($LASTEXITCODE -ne 0 -or $sharedAfter.Id -ne $shared.Id -or $sharedAfter.State.StartedAt -ne $shared.State.StartedAt) {
        throw '2026 컨테이너의 연속성이 달라졌습니다.'
    }
    $report.renamed = $true
}
catch {
    $report.error = $_.Exception.ToString()
    throw
}
finally {
    $report.finished_kst = Get-Date -Format o
    $logRoot = if (Test-Path -LiteralPath $target) { $target } else { $source }
    $report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $logRoot 'evidence\migration.json') -Encoding utf8
    $report | ConvertTo-Json -Depth 5
}
