#!/usr/bin/env pwsh
#MISE description="Start reranker server — bge-reranker-v2-m3 cross-encoder (:8082) with logging"
#MISE alias="sr"
#MISE depends=["ensure-logs-dir"]

param(
    [string]$Model = 'gpustack/bge-reranker-v2-m3-GGUF:Q4_K_M',
    # -Ctx is capped by the model: llama-server clamps the slot ctx to the
    # model's training ctx (8192) regardless of the requested -c, so 32768
    # above already allocates 8192. vram_sweep confirmed ctx variants are
    # byte-identical (all 309 MB) — only -Batch/-ubatch are a real VRAM lever,
    # and lowering batch to 4096 measured +134 MB (worse). Both left as-is.
    [int]$Ctx = 32768,
    [int]$Batch = 8192
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
    $serverArgs = @('-m',$Model,'--alias','bge-reranker-v2-m3','--host','0.0.0.0','--port','8082','--reranking','--pooling','rank','-c',"$Ctx",'-b',"$Batch",'-ub',"$Batch",'-ngl','99',"--log-file","$logFile")
} else {
    $serverArgs = @('-hf',$Model,'--alias','bge-reranker-v2-m3','--host','0.0.0.0','--port','8082','--reranking','--pooling','rank','-c',"$Ctx",'-b',"$Batch",'-ub',"$Batch",'-ngl','99',"--log-file","$logFile")
}

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started reranker server (:8082) - log: $logFile"
