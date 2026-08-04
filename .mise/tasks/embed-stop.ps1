#!/usr/bin/env pwsh
#MISE description="Stop embedding server (kills process on port 8081)"
#MISE alias="xe"

param()

# Try by port first
$p = $null
try {
    $portProc = Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction Stop
    if ($portProc) {
        $p = Get-Process -Id $portProc.OwningProcess -ErrorAction Stop
    }
} catch { }

# Fallback: find llama-server on port 8081
if (-not $p) {
    $p = Get-Process -Name llama-server -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_ | Get-NetTCPConnection -LocalPort 8081 -ErrorAction Stop
            $true
        } catch { $false }
    }
}

if ($p) {
    Stop-Process $p -Force
    "Stopped embed"
} else {
    "Not running"
}