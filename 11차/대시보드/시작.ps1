Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$python = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw '기존 Codex runtime Python이 없습니다. 기동 담당자가 실행 환경을 확인해야 합니다.'
}

# 부모가 기존 읽기 전용 화면을 교체한 뒤 이 E11 경로에서만 시작합니다.
$exitCode = 1
Push-Location -LiteralPath $PSScriptRoot
try {
    & $python -B -X utf8 (Join-Path $PSScriptRoot 'server.py') --port 8881
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $exitCode
