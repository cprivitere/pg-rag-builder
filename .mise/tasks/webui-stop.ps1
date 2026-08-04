#!/usr/bin/env pwsh
#MISE description="Stop Open WebUI (kills process on port 3000)"
#MISE alias="xw"

param()

$p = $null
try {
    $portProc = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction Stop
    if ($portProc) {
        $p = Get-Process -Id $portProc.OwningProcess -ErrorAction Stop
    }
} catch { }

if (-not $p) {
    $p = Get-Process -Name uv -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_ | Get-NetTCPConnection -LocalPort 3000 -ErrorAction Stop
            $true
        } catch { $false }
    }
}

if (-not $p) {
    $p = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        try {
            $_ | Get-NetTCPConnection -LocalPort 3000 -ErrorAction Stop
            $true
        } catch { $false }
    }
}

if ($p) {
    Stop-Process $p -Force
    "Stopped WebUI"
} else {
    "Not running"
}