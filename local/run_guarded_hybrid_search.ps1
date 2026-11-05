param([Parameter(ValueFromRemainingArguments = $true)][string[]]$RunArguments)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$candidates = @(
    (Join-Path $env:ProgramFiles 'Git/bin/bash.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs/Git/bin/bash.exe')
)
$gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
if ($gitCommand) {
    $gitRoot = Split-Path -Parent (Split-Path -Parent $gitCommand.Source)
    $candidates += Join-Path $gitRoot 'bin/bash.exe'
}
$gitBash = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $gitBash) { throw 'Git Bash not found.' }
Push-Location -LiteralPath $projectRoot
try {
    & $gitBash 'local/run_guarded_hybrid_search.sh' @RunArguments
    $runExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $runExitCode
