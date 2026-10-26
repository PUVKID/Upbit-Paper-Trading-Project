param(
    [ValidateSet('start', 'status', 'stop', 'gpu', 'test', 'serve')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
$runtimePython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) {
    throw '검증된 파이썬 실행 파일이 없습니다. 도커 관리 도구의 실행 경로를 확인하십시오.'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $runtimePython -B -X utf8 (Join-Path $PSScriptRoot 'tools\docker_control.py') $Action
    if ($LASTEXITCODE -ne 0) { throw "도커 관리 명령이 실패했습니다. 종료 코드: $LASTEXITCODE" }
}
finally {
    Pop-Location
}
