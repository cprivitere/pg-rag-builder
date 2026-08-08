#!/usr/bin/env pwsh
#MISE description="Start reranker server — Qwen3-Reranker-0.6B cross-encoder (:8082) with logging"
#MISE alias="sr"
#MISE depends=["ensure-logs-dir"]

param(
    [string]$Model = 'gpustack/bge-reranker-v2-m3-GGUF:Q4_K_M',
    [int]$Ctx = 32768
)

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort 8082 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Reranker server already running on :8082 (PID $($existing.OwningProcess))"
    exit 0
}

$exe = 'llama-server'
$logFile = Join-Path $logDir 'rerank.log'
$target = (Split-Path -Parent $Model) + [IO.Path]::DirectorySeparatorChar + (Split-Path -Leaf $Model)
if (Test-Path -LiteralPath $Model) {
    $serverArgs = @('-m',$Model,'--alias','qwen3-reranker','--host','0.0.0.0','--port','8082','--reranking','--pooling','rank','-c',"$Ctx",'-ngl','99',"--log-file","$logFile")
} else {
    $serverArgs = @('-hf',$Model,'--alias','qwen3-reranker','--host','0.0.0.0','--port','8082','--reranking','--pooling','rank','-c',"$Ctx",'-ngl','99',"--log-file","$logFile")
}

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started reranker server (:8082) - log: $logFile"