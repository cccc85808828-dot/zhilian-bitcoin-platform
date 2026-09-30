$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$serverScript = Join-Path $projectRoot 'portable_offline_server.ps1'
if ($env:ZHILIAN_NO_BROWSER -eq '1') {
    & $serverScript -NoBrowser
}
else {
    & $serverScript
}
