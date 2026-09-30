$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    & $python.Source 'scripts\run_competition_demo.py'
    exit $LASTEXITCODE
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($pyLauncher) {
    & $pyLauncher.Source -3 'scripts\run_competition_demo.py'
    exit $LASTEXITCODE
}

throw 'Python was not found. Create the environment described in environment.yml first.'
