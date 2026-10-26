param([Parameter(ValueFromRemainingArguments = $true)][string[]]$RunArguments)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_feature_ablation.ps1') --env-file .env.deepseek @RunArguments
exit $LASTEXITCODE
