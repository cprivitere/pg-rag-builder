#!/usr/bin/env pwsh
#MISE description="Stop reranker server (kills process on port 8082)"
#MISE alias="xr"

param()

$p = $null
try {
    $portProc = Get-NetTCPConnection -LocalPort 8082 -State Listen -ErrorAction Stop
    if ($portProc) {
        $p = Get-Process -Id $portProc.OwningProcess -ErrorAction Stop
    }
} catch { }

if (-not $p) {
    $p = Get-Process -Name llama-server -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_ | Get-NetTCPConnection -LocalPort 8082 -ErrorAction Stop
            $true
        } catch { $false }
    }
}

if ($p) {
    Stop-Process $p -Force
    "Stopped rerank"
} else {
    "Not running"
}