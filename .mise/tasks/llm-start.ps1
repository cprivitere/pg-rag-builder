#!/usr/bin/env pwsh
#MISE description="Start LLM server (gemma-4-26B on :8080) with logging"
#MISE alias="sl"
#MISE depends=["ensure-logs-dir"]

param()

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$exe = 'llama-server'
$serverArgs = @('-hf','unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL','--spec-type','draft-mtp','--spec-draft-n-max','4','-ngl','999','-fa','on','-c','16384','--reasoning-budget','1024','-ctk','q8_0','-ctv','q8_0','--host','0.0.0.0','--port','8080')
$outLog = Join-Path $logDir 'llm.log'
$errLog = Join-Path $logDir 'llm-error.log'

$batPath = [IO.Path]::GetTempFileName() + '.bat'
$batContent = "@echo off`n`"$exe`" $($serverArgs -join ' ') > `"$outLog`" 2> `"$errLog`""
[IO.File]::WriteAllText($batPath, $batContent)

Start-Process -FilePath 'cmd.exe' -ArgumentList "/c `"$batPath`"" -WindowStyle Hidden

Write-Host "Started LLM server (:8080) - logs in logs/llm.log"