#!/usr/bin/env pwsh
#MISE description="Stop LLM server (kills process on port 8080)"
#MISE alias="xl"

param()

$p = $null
try {
    $portProc = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction Stop
    if ($portProc) {
        $p = Get-Process -Id $portProc.OwningProcess -ErrorAction Stop
    }
} catch { }

if (-not $p) {
    $p = Get-Process -Name llama-server -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_ | Get-NetTCPConnection -LocalPort 8080 -ErrorAction Stop
            $true
        } catch { $false }
    }
}

if ($p) {
    Stop-Process $p -Force
    "Stopped LLM"
} else {
    "Not running"
}