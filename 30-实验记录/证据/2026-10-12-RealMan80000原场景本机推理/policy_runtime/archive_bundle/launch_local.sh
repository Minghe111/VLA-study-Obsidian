#!/usr/bin/env bash
set -euo pipefail
task_runtime=/tmp/realman-eval80k-20261012/policy_runtime
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export JAX_COMPILATION_CACHE_DIR="$task_runtime/jax_cache"
export PATH="$task_runtime/venv/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$task_runtime/venv/lib/python3.12/site-packages/nvidia/cuda_nvcc"
unset LD_LIBRARY_PATH
exec "$task_runtime/venv/bin/python" "$task_runtime/service80k.py" "$@"
