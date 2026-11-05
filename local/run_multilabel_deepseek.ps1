# Validate the frozen classifier before opening SSH; paid requests require manual launch.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$RunArguments)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
$benchmarkArguments = @(
    '--configuration', 'multilabel',
    '--classifier-model', 'research_results/multilabel_classifier_deepseek_20261029',
    '--env-file', '.env.deepseek', '--model', 'deepseek-flash',
    '--benchmark', 'both', '--classifier-split', 'test', '--evaluation-policy', 'strict',
    '--output', 'research_results/multilabel_deepseek_test_20261029'
) + $RunArguments
Push-Location -LiteralPath $projectRoot
try {
    & $python 'scripts/run_openrouter_baseline.py' @benchmarkArguments '--dry-run'
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($RunArguments -contains '--dry-run' -or $RunArguments -contains '--help') { exit 0 }
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
    & $gitBash 'local/run_openrouter_baseline.sh' @benchmarkArguments
    $runExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $runExitCode
