# Geniee local Python troubleshooting pipeline.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project interpreter not found: $python"
}

Set-Location $root
& $python data\build_conversation_dataset.py
& $python data\inspect_conversation_dataset.py
& $python data\build_instruction_dataset.py
& $python training\train_sft.py
& $python evaluation\evaluate_sft.py --checkpoint checkpoints\geniee_sft_best.pt --greedy --limit 24
& $python -m pytest tests\test_chat.py tests\test_generation.py tests\test_tokenizer.py -q

Write-Host "Geniee Python troubleshooting pipeline completed." -ForegroundColor Green