# Same training budget as default (500 physics pretrain + 4000 joint, full CSV + AUGMENT_PER_ROW).
# Only enables CPU/runtime optimizations: PyTorch threads, larger joint mini-batches, parallel ODE prep.
# Shorter runs: set PINN_MEDIUM_TRAIN=1 or PINN_QUICK_TRAIN=1 yourself (not recommended for final weights).

$env:PINN_FAST_CPU = "1"
# Optional — faster per-epoch joint step if you have enough RAM (~16GB+ often OK):
# $env:PINN_JOINT_CHUNK = "0"

& "$PSScriptRoot\run_train.ps1" @args
