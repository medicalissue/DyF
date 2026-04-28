#!/usr/bin/env bash
# Smoke-test: build a tiny fake ImageFolder dataset, run 2 epochs through
# main.py for every activation variant. No GPU required (CPU is fine, slow).
#
# Run from repo root:
#     bash scripts/dryrun.sh
#
# This validates the full path:
#   YAML → yaml_to_args → main.py → convert_ln_to_dyf → train loop
# without touching real ImageNet.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-/tmp/dyf_dryrun_data}"
OUT_ROOT="${OUT_ROOT:-/tmp/dyf_dryrun_out}"
ACTIVATIONS=("${@:-ln dyt dyf dyf-hard}")

# 1) Build fake dataset (10 classes × 4 images each).
if [[ ! -d "$DATA_DIR/train/cls0" ]]; then
    echo "▶ creating fake dataset at $DATA_DIR"
    python -c "
import os, numpy as np
from PIL import Image
root = '$DATA_DIR'
for split in ('train', 'val'):
    for c in range(10):
        d = f'{root}/{split}/cls{c}'
        os.makedirs(d, exist_ok=True)
        for i in range(4):
            img = (np.random.rand(64, 64, 3) * 255).astype('uint8')
            Image.fromarray(img).save(f'{d}/{i}.png')
print('done')
"
fi

# 2) Source activation→flags mapping.
# shellcheck source=infra/activation_flags.sh
source "$REPO_ROOT/scripts/infra/activation_flags.sh"

# 3) Run each activation variant.
for act in $ACTIVATIONS; do
    out="$OUT_ROOT/$act"
    mkdir -p "$out"
    echo
    echo "════════════════════════════════════════════════════════"
    echo "  dryrun: activation=$act  out=$out"
    echo "════════════════════════════════════════════════════════"

    cfg_args=$(python "$REPO_ROOT/scripts/infra/yaml_to_args.py" \
               "$REPO_ROOT/configs/dryrun/toy.yaml")
    act_args=$(activation_flags "$act")

    # CPU single-process: skip torchrun, run main.py directly. main.py
    # calls init_distributed_mode which short-circuits sensibly when
    # WORLD_SIZE/RANK env vars are unset.
    # shellcheck disable=SC2086
    python "$REPO_ROOT/main.py" \
        $cfg_args $act_args \
        --data_path "$DATA_DIR" \
        --output_dir "$out" \
        --device "${DEVICE:-cpu}" \
        2>&1 | tail -40
done

echo
echo "✓ dryrun complete. Each activation should have logged 2 epochs."
