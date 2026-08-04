#!/usr/bin/env pwsh
#MISE description="Start embedding server — production (4 slots, 8k context) with logging"
#MISE alias="se"
#MISE depends=["ensure-logs-dir"]

param()

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$exe = 'llama-server'
$serverArgs = @('-hf','jinaai/jina-embeddings-v5-text-small-retrieval-GGUF:Q8_0','--host','0.0.0.0','--port','8081','--embedding','--pooling','last','-ngl','99','-np','4','-c','8192')
$outLog = Join-Path $logDir 'embed.log'
$errLog = Join-Path $logDir 'embed-error.log'

# Write a temp batch file that handles redirection, then launch it detached
$batPath = [IO.Path]::GetTempFileName() + '.bat'
$batContent = "@echo off`n`"$exe`" $($serverArgs -join ' ') > `"$outLog`" 2> `"$errLog`""
[IO.File]::WriteAllText($batPath, $batContent)

# Launch batch file hidden, no redirection - exits immediately
Start-Process -FilePath 'cmd.exe' -ArgumentList "/c `"$batPath`"" -WindowStyle Hidden

Write-Host "Started embedding server (:8081) - logs in logs/embed.log"