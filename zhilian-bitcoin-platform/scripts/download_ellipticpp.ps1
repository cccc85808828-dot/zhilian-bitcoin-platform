param(
    [string]$RepoPath = "data/raw/ellipticpp"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repository = (Resolve-Path -LiteralPath $RepoPath).Path
if (-not (Test-Path -LiteralPath (Join-Path $repository ".git"))) {
    throw "Elliptic++ Git repository was not found at: $repository"
}

$endpoint = "https://github.com/git-disl/EllipticPlusPlus.git/info/lfs/objects/batch"
$objects = @(
    @{ Path = "Actors Dataset/AddrAddr_edgelist.csv"; Oid = "ffba894458e262a691e5e4d006f5dc1d0e069fabfe828f443fd157bf7f8393f2"; Size = 200631481L },
    @{ Path = "Actors Dataset/AddrTx_edgelist.csv"; Oid = "f5f903f752387f66a1bccaeff54e293b2e8470fcddf5eb56b88aa06fd23a8f3b"; Size = 21248388L },
    @{ Path = "Actors Dataset/TxAddr_edgelist.csv"; Oid = "9f5afbdde7bc3d91fb7a4655be55799d6504cd0063ae55a0753a5a41189932b8"; Size = 36702878L },
    @{ Path = "Actors Dataset/wallets_classes.csv"; Oid = "4e5132c99f941666bf1fefd4100a1428d339c9252ec6987909e1adf8eac902f9"; Size = 30421134L },
    @{ Path = "Actors Dataset/wallets_features.csv"; Oid = "317daca2810c355ddfdb8c0dab34cf11d1aa90567fe975090c7e5a901386eb77"; Size = 606463522L },
    @{ Path = "Actors Dataset/wallets_features_classes_combined.csv"; Oid = "99bf27f7b76d6578ad59e0a61ec225ecd656fef6ae6958f29435ab286be2cc7d"; Size = 609000048L },
    @{ Path = "Transactions Dataset/txs_classes.csv"; Oid = "013a11742969071a906878ded0319571df0657f9b7133e5c6cdb36217bf0d240"; Size = 2361914L },
    @{ Path = "Transactions Dataset/txs_edgelist.csv"; Oid = "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e"; Size = 4470584L },
    @{ Path = "Transactions Dataset/txs_features.csv"; Oid = "2db326ec8ddb68f1d810c1834e1ff62e0a8300378f0984a1e3b2ca82a439821b"; Size = 694789588L }
)

function Get-DownloadUrl {
    param([string]$Oid, [long]$Size)

    $body = @{
        operation = "download"
        transfers = @("basic")
        objects = @(@{ oid = $Oid; size = $Size })
    } | ConvertTo-Json -Depth 5 -Compress

    $response = Invoke-RestMethod `
        -Uri $endpoint `
        -Method Post `
        -Headers @{ Accept = "application/vnd.git-lfs+json" } `
        -ContentType "application/vnd.git-lfs+json" `
        -Body $body `
        -TimeoutSec 30
    return $response.objects[0].actions.download.href
}

foreach ($object in $objects) {
    $oid = [string]$object.Oid
    $size = [long]$object.Size
    $objectDirectory = Join-Path $repository (".git/lfs/objects/{0}/{1}" -f $oid.Substring(0, 2), $oid.Substring(2, 2))
    $objectPath = Join-Path $objectDirectory $oid
    $partialPath = Join-Path $repository (".git/lfs/incomplete/$oid")

    if ((Test-Path -LiteralPath $objectPath) -and (Get-Item -LiteralPath $objectPath).Length -eq $size) {
        Write-Output ("[skip] {0}" -f $object.Path)
        continue
    }

    $verified = $false
    if ((Test-Path -LiteralPath $partialPath) -and (Get-Item -LiteralPath $partialPath).Length -eq $size) {
        $partialHash = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $verified = $partialHash -eq $oid
    }

    for ($attempt = 1; -not $verified -and $attempt -le 10; $attempt++) {
        Write-Output ("[download {0}/10] {1}" -f $attempt, $object.Path)
        $url = Get-DownloadUrl -Oid $oid -Size $size
        & curl.exe `
            -L `
            -C - `
            --fail `
            --silent `
            --show-error `
            --retry 5 `
            --retry-delay 3 `
            --retry-all-errors `
            --connect-timeout 30 `
            --max-time 900 `
            --speed-time 120 `
            --speed-limit 1024 `
            --output $partialPath `
            $url

        if ($LASTEXITCODE -ne 0) {
            continue
        }
        if ((Get-Item -LiteralPath $partialPath).Length -ne $size) {
            continue
        }
        $partialHash = (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $verified = $partialHash -eq $oid
    }

    if (-not $verified) {
        throw "Unable to download and verify: $($object.Path)"
    }

    if (-not (Test-Path -LiteralPath $objectDirectory)) {
        New-Item -ItemType Directory -Path $objectDirectory -Force | Out-Null
    }
    if (Test-Path -LiteralPath $objectPath) {
        [System.IO.File]::Delete($objectPath)
    }
    [System.IO.File]::Move($partialPath, $objectPath)
    Write-Output ("[verified] {0} ({1:N2} MB)" -f $object.Path, ($size / 1MB))
}

& git -C $repository lfs checkout
if ($LASTEXITCODE -ne 0) {
    throw "git lfs checkout failed."
}

Write-Output "Elliptic++ download and LFS checkout completed."
