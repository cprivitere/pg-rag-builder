#!/usr/bin/env pwsh
#MISE description="Start embedding server — production (4 slots, 8k context) with logging"
#MISE alias="se"
#MISE depends=["ensure-logs-dir"]

param()

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Embedding server already running on :8081 (PID $($existing.OwningProcess))"
    exit 0
}

$exe = 'llama-server'
$logFile = Join-Path $logDir 'embed.log'
$serverArgs = @('-hf','jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0','--host','0.0.0.0','--port','8081','--embedding','--pooling','last','-ngl','99','-np','4','-c','8192',"--log-file","$logFile")

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started embedding server (:8081) - log: $logFile"