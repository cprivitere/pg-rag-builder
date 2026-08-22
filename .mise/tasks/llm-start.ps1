#!/usr/bin/env pwsh
#MISE description="Start LLM server (gemma-4-26B on :8080) with logging"
#MISE alias="sl"
#MISE depends=["ensure-logs-dir"]

param()

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "LLM server already running on :8080 (PID $($existing.OwningProcess))"
    exit 0
}

$exe = 'llama-server'
$logFile = Join-Path $logDir 'llm.log'
# Speculative decoding (--spec-type draft-mtp) ON: costs +323 MB committed VRAM
# (~2%) but a high-N benchmark (30 samples, interleaved rounds, 256-token
# prompt-cached decode) measured MTP at 155 vs 98 tok/s = 1.58x faster decode —
# a decisive latency win for a single-user RAG chat pipe, worth the 2% VRAM.
# (Early low-N runs misleadingly suggested no-spec was faster/neutral.) ctx
# stays 16384: shrinking saves only ~0.1 GB (KV tiny vs weights) and risks
# truncating the RAG window (CONTEXT_BUDGET ~ 12-13k tokens).
$serverArgs = @('-hf','unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL','--spec-type','draft-mtp','--spec-draft-n-max','4','-ngl','999','-fa','on','-c','16384','-np','1','--reasoning-budget','1024','--host','0.0.0.0','--port','8080',"--log-file","$logFile")

Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden

Write-Host "Started LLM server (:8080) - log: $logFile"