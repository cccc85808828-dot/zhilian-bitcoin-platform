param(
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$demoRoot = Join-Path $projectRoot 'demo'
$fixturePath = Join-Path $demoRoot 'data\offline_fixtures.json'

if (-not (Test-Path -LiteralPath $fixturePath)) {
    throw 'Offline fixture file is missing.'
}

$fixtures = Get-Content -LiteralPath $fixturePath -Raw -Encoding UTF8 | ConvertFrom-Json
$utf8 = New-Object System.Text.UTF8Encoding($false)
$ascii = [System.Text.Encoding]::ASCII

function Get-MapValue {
    param($Map, [string]$Name)
    if ($null -eq $Map) { return $null }
    $property = $Map.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

function Get-PropertyCount {
    param($Map)
    if ($null -eq $Map) { return 0 }
    return @($Map.PSObject.Properties).Count
}

function Find-HeaderEnd {
    param([byte[]]$Bytes)
    for ($i = 0; $i -le $Bytes.Length - 4; $i++) {
        if ($Bytes[$i] -eq 13 -and $Bytes[$i + 1] -eq 10 -and
            $Bytes[$i + 2] -eq 13 -and $Bytes[$i + 3] -eq 10) {
            return $i
        }
    }
    return -1
}

function Read-HttpRequest {
    param([System.Net.Sockets.TcpClient]$Client)
    $stream = $Client.GetStream()
    $stream.ReadTimeout = 10000
    $memory = New-Object System.IO.MemoryStream
    $buffer = New-Object byte[] 8192
    $headerEnd = -1

    while ($headerEnd -lt 0) {
        $count = $stream.Read($buffer, 0, $buffer.Length)
        if ($count -le 0) { break }
        $memory.Write($buffer, 0, $count)
        if ($memory.Length -gt 65536) { throw 'HTTP header is too large.' }
        $headerEnd = Find-HeaderEnd -Bytes $memory.ToArray()
    }
    if ($headerEnd -lt 0) { throw 'Incomplete HTTP request.' }

    $allBytes = $memory.ToArray()
    $headerText = $ascii.GetString($allBytes, 0, $headerEnd)
    $lines = $headerText -split "`r`n"
    $requestParts = $lines[0] -split ' '
    if ($requestParts.Count -lt 2) { throw 'Invalid HTTP request line.' }

    $headers = @{}
    for ($i = 1; $i -lt $lines.Count; $i++) {
        $separator = $lines[$i].IndexOf(':')
        if ($separator -gt 0) {
            $name = $lines[$i].Substring(0, $separator).Trim().ToLowerInvariant()
            $value = $lines[$i].Substring($separator + 1).Trim()
            $headers[$name] = $value
        }
    }

    $contentLength = 0
    if ($headers.ContainsKey('content-length')) {
        $contentLength = [int]$headers['content-length']
    }
    if ($contentLength -gt 1048576) { throw 'HTTP body is too large.' }

    $bodyStart = $headerEnd + 4
    while (($allBytes.Length - $bodyStart) -lt $contentLength) {
        $count = $stream.Read($buffer, 0, $buffer.Length)
        if ($count -le 0) { break }
        $memory.Position = $memory.Length
        $memory.Write($buffer, 0, $count)
        $allBytes = $memory.ToArray()
    }
    if (($allBytes.Length - $bodyStart) -lt $contentLength) {
        throw 'Incomplete HTTP body.'
    }

    $body = ''
    if ($contentLength -gt 0) {
        $body = $utf8.GetString($allBytes, $bodyStart, $contentLength)
    }
    return [pscustomobject]@{
        Method = $requestParts[0].ToUpperInvariant()
        Target = $requestParts[1]
        Headers = $headers
        Body = $body
        Stream = $stream
    }
}

function Write-HttpResponse {
    param(
        [System.IO.Stream]$Stream,
        [int]$StatusCode,
        [string]$StatusText,
        [string]$ContentType,
        [byte[]]$Content
    )
    $header = "HTTP/1.1 $StatusCode $StatusText`r`n" +
        "Content-Type: $ContentType`r`n" +
        "Content-Length: $($Content.Length)`r`n" +
        "Cache-Control: no-store`r`n" +
        "X-Content-Type-Options: nosniff`r`n" +
        "Connection: close`r`n`r`n"
    $headerBytes = $ascii.GetBytes($header)
    $Stream.Write($headerBytes, 0, $headerBytes.Length)
    if ($Content.Length -gt 0) {
        $Stream.Write($Content, 0, $Content.Length)
    }
    $Stream.Flush()
}

function Write-JsonResponse {
    param(
        [System.IO.Stream]$Stream,
        $Payload,
        [int]$StatusCode = 200,
        [string]$StatusText = 'OK'
    )
    $json = $Payload | ConvertTo-Json -Depth 100 -Compress
    Write-HttpResponse -Stream $Stream -StatusCode $StatusCode -StatusText $StatusText `
        -ContentType 'application/json; charset=utf-8' -Content $utf8.GetBytes($json)
}

function Parse-Query {
    param([string]$Query)
    $values = @{}
    $text = $Query.TrimStart('?')
    if ([string]::IsNullOrWhiteSpace($text)) { return $values }
    foreach ($part in ($text -split '&')) {
        $pair = $part -split '=', 2
        $name = [System.Uri]::UnescapeDataString(($pair[0] -replace '\+', ' '))
        $value = ''
        if ($pair.Count -gt 1) {
            $value = [System.Uri]::UnescapeDataString(($pair[1] -replace '\+', ' '))
        }
        $values[$name] = $value
    }
    return $values
}

function Build-AssistantAnswer {
    param([string]$Question, [string]$TransactionId, [string]$Address)
    $ui = $fixtures.ui
    $analysis = Get-MapValue -Map $fixtures.analyses -Name $TransactionId
    $profile = Get-MapValue -Map $fixtures.profiles -Name $Address
    $technical = $Question -match '(?i)TTHGNN|QIF|HGNN|model|method'

    if ($technical) {
        $answer = [string]$ui.assistant_technical
        if ($null -ne $analysis) {
            $explanation = $analysis.result.decision_explanation
            $answer += "`n`n" + [string]$explanation.structure.summary
            $answer += "`n`n" + [string]$explanation.attribute.summary
            $answer += "`n`n" + [string]$explanation.fusion.summary
        }
        return $answer
    }

    if ($null -ne $analysis) {
        $result = $analysis.result
        $explanation = $result.decision_explanation
        $heading = $ui.assistant_headings
        return (
            "## $($heading.object)`n" +
            "$($result.transaction_id)`n`n" +
            "## $($heading.behavior)`n$($explanation.behavior.summary)`n`n" +
            "## $($heading.structure)`n$($explanation.structure.summary)`n`n" +
            "## $($heading.attribute)`n$($explanation.attribute.summary)`n`n" +
            "## $($heading.fusion)`n$($explanation.fusion.summary)`n`n" +
            "## $($heading.conclusion)`n$($explanation.conclusion)"
        )
    }

    if ($null -ne $profile) {
        $graph = $profile.relation_graph
        return ([string]$ui.assistant_address_template -f
            $profile.related_transaction_count,
            $profile.high_risk_transactions,
            $graph.node_count,
            $graph.edge_count)
    }
    return [string]$ui.assistant_default
}

function Send-StaticFile {
    param([System.IO.Stream]$Stream, [string]$Path)
    $allowed = @{
        '/' = 'index.html'
        '/index.html' = 'index.html'
        '/relations.html' = 'relations.html'
        '/analysis.html' = 'analysis.html'
        '/styles.css' = 'styles.css'
        '/common.js' = 'common.js'
        '/address.js' = 'address.js'
        '/relations.js' = 'relations.js'
        '/analysis.js' = 'analysis.js'
    }
    if (-not $allowed.ContainsKey($Path)) {
        Write-JsonResponse -Stream $Stream -Payload @{ error = 'Resource not found.' } `
            -StatusCode 404 -StatusText 'Not Found'
        return
    }
    $fileName = $allowed[$Path]
    $filePath = Join-Path $demoRoot $fileName
    $mime = 'application/octet-stream'
    if ($fileName.EndsWith('.html')) { $mime = 'text/html; charset=utf-8' }
    elseif ($fileName.EndsWith('.css')) { $mime = 'text/css; charset=utf-8' }
    elseif ($fileName.EndsWith('.js')) { $mime = 'application/javascript; charset=utf-8' }
    $content = [System.IO.File]::ReadAllBytes($filePath)
    Write-HttpResponse -Stream $Stream -StatusCode 200 -StatusText 'OK' `
        -ContentType $mime -Content $content
}

function Handle-Request {
    param($Request)
    $uri = New-Object System.Uri("http://127.0.0.1$($Request.Target)")
    $path = $uri.AbsolutePath

    if ($Request.Method -eq 'GET') {
        if ($path -eq '/api/health') {
            $payload = @{
                status = 'ok'
                method = $fixtures.metadata.model
                transactions = Get-PropertyCount $fixtures.cases
                indexed_addresses = Get-PropertyCount $fixtures.profiles
                assistant_mode = 'offline_knowledge'
                data_mode = 'offline_snapshot'
                chain_data = @{
                    enabled = $false
                    network = $fixtures.metadata.network
                    provider = $fixtures.metadata.provider
                    captured_at = $fixtures.metadata.captured_at
                    model_loaded = $true
                }
            }
            Write-JsonResponse -Stream $Request.Stream -Payload $payload
            return
        }
        if ($path -eq '/api/summary') {
            Write-JsonResponse -Stream $Request.Stream -Payload $fixtures.summary
            return
        }
        if ($path -eq '/api/address-samples') {
            $samples = @($fixtures.samples | ForEach-Object {
                @{ label = $_.label; address = $_.address; description = $_.description }
            })
            Write-JsonResponse -Stream $Request.Stream -Payload $samples
            return
        }
        if ($path -eq '/api/assistant/status') {
            Write-JsonResponse -Stream $Request.Stream -Payload @{
                mode = 'offline_knowledge'; model = $null; configured = $false
            }
            return
        }
        if ($path -eq '/api/address') {
            $query = Parse-Query $uri.Query
            $profile = Get-MapValue -Map $fixtures.profiles -Name ([string]$query['value'])
            if ($null -eq $profile) {
                Write-JsonResponse -Stream $Request.Stream -Payload @{
                    error = $fixtures.ui.offline_address_error
                    hint = $fixtures.ui.offline_address_hint
                } -StatusCode 404 -StatusText 'Not Found'
            } else {
                Write-JsonResponse -Stream $Request.Stream -Payload $profile
            }
            return
        }
        if ($path.StartsWith('/api/case/')) {
            $transactionId = [System.Uri]::UnescapeDataString($path.Substring(10))
            $case = Get-MapValue -Map $fixtures.cases -Name $transactionId
            if ($null -eq $case) {
                Write-JsonResponse -Stream $Request.Stream -Payload @{ error = 'Transaction not found.' } `
                    -StatusCode 404 -StatusText 'Not Found'
            } else {
                Write-JsonResponse -Stream $Request.Stream -Payload $case
            }
            return
        }
        if ($path.StartsWith('/api/')) {
            Write-JsonResponse -Stream $Request.Stream -Payload @{ error = 'API not found.' } `
                -StatusCode 404 -StatusText 'Not Found'
            return
        }
        Send-StaticFile -Stream $Request.Stream -Path $path
        return
    }

    if ($Request.Method -eq 'POST' -and ($path -eq '/api/analyze' -or $path -eq '/api/assistant')) {
        try {
            $body = $Request.Body | ConvertFrom-Json
        } catch {
            Write-JsonResponse -Stream $Request.Stream -Payload @{ error = 'Invalid JSON body.' } `
                -StatusCode 400 -StatusText 'Bad Request'
            return
        }
        if ($path -eq '/api/analyze') {
            $transactionId = [string]$body.transaction_id
            $analysis = Get-MapValue -Map $fixtures.analyses -Name $transactionId
            if ($null -eq $analysis) {
                Write-JsonResponse -Stream $Request.Stream -Payload @{ error = 'Transaction not found.' } `
                    -StatusCode 404 -StatusText 'Not Found'
            } else {
                Write-JsonResponse -Stream $Request.Stream -Payload $analysis
            }
            return
        }
        $question = [string]$body.question
        if ([string]::IsNullOrWhiteSpace($question)) {
            Write-JsonResponse -Stream $Request.Stream -Payload @{ error = 'Question is empty.' } `
                -StatusCode 400 -StatusText 'Bad Request'
            return
        }
        $answer = Build-AssistantAnswer -Question $question `
            -TransactionId ([string]$body.transaction_id) -Address ([string]$body.address)
        Write-JsonResponse -Stream $Request.Stream -Payload @{
            status = 'PASS'; mode = 'offline_knowledge'; model = $null; answer = $answer
        }
        return
    }

    Write-JsonResponse -Stream $Request.Stream -Payload @{ error = 'Method not allowed.' } `
        -StatusCode 405 -StatusText 'Method Not Allowed'
}

$listener = $null
$selectedPort = $Port
for ($candidate = $Port; $candidate -le ($Port + 10); $candidate++) {
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback, $candidate
        )
        $listener.Start()
        $selectedPort = $candidate
        break
    } catch {
        $listener = $null
    }
}
if ($null -eq $listener) { throw 'No local TCP port is available.' }

$url = "http://127.0.0.1:$selectedPort/"
Write-Host "Zhilian offline demo is ready: $url"
Write-Host 'Press Ctrl+C to stop.'
if (-not $NoBrowser) {
    Start-Process $url
}

try {
    while ($true) {
        $client = $listener.AcceptTcpClient()
        $client.NoDelay = $true
        try {
            $request = Read-HttpRequest -Client $client
            Handle-Request -Request $request
        } catch {
            try {
                $stream = $client.GetStream()
                Write-JsonResponse -Stream $stream -Payload @{ error = $_.Exception.Message } `
                    -StatusCode 500 -StatusText 'Internal Server Error'
            } catch {}
        } finally {
            $client.Close()
        }
    }
} finally {
    $listener.Stop()
}
