#!/usr/bin/env pwsh
#MISE description="Start embedding server — build mode (1 slot, 64k context) with logging"
#MISE alias="eb"
#MISE depends=["ensure-logs-dir"]

param()

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$exe = 'llama-server'
$logFile = Join-Path $logDir 'embed-build.log'
$serverArgs = @('-hf','jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0','--host','0.0.0.0','--port','8081','--embedding','--pooling','last','-ngl','99','-np','1','-c','64000',"--log-file","$logFile")

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started embedding server build mode (:8081) - log: $logFile"