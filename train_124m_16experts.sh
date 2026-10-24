#!/usr/bin/env bash
# Original native DeepSeek trainer; 16 routed experts, top-2, ~124M parameters.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
data_root=${1:-edu_fineweb10B}
if [[ $# -gt 0 ]]; then shift; fi
architecture=(--config "$project_root/configs/deepseek_124m_16experts.json")
for arg in "$@"; do
  if [[ "$arg" == --resume || "$arg" == --resume=* ]]; then architecture=(); fi
done
exec python -u train_deepseek_v3.py \
  "${architecture[@]}" \
  --data-root "$data_root" --log-dir log_124m_16experts \
  --device cuda --batch-size 8 --seq-len 1024 \
  --total-batch-size 524288 --max-steps 19073 \
  --max-lr 0.0006 --warmup-steps 715 \
  --val-interval 250 --val-steps 10 \
  --auto-restart --max-restarts 5 --restart-delay 15 \
  --save-interval 25 --keep-checkpoints 3 "$@"
