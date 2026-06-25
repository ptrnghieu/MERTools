#!/bin/bash
# Setup script for new Vast.ai instance
# Usage: bash setup-instance.sh <HF_TOKEN> <DATA_ROOT>
# Example: bash setup-instance.sh hf_xxxx /root/data

set -e

HF_TOKEN="${1:?Usage: bash setup-instance.sh <HF_TOKEN> <DATA_ROOT>}"
DATA_ROOT="${2:?Usage: bash setup-instance.sh <HF_TOKEN> <DATA_ROOT>}"

FEATURES_DIR="$DATA_ROOT/mer2026-dataset-process"
mkdir -p "$FEATURES_DIR"

echo "=== Step 1: Install dependencies ==="
pip install -q fire huggingface_hub

echo "=== Step 2: Download pre-extracted features ==="
python3 - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="hhieupt/mer2026-features",
    repo_type="dataset",
    token="$HF_TOKEN",
    local_dir="$FEATURES_DIR",
)
print("Features downloaded OK")
PYEOF

echo "=== Step 3: Download CSV metadata from official MER2026 repo ==="
python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import os

files = [
    "track1_train.csv",
    "track1_track2_candidate.csv",
    "track2_train_human.csv",
]
for f in files:
    dest = os.path.join("$FEATURES_DIR", f)
    if os.path.exists(dest):
        print(f"  {f} already exists, skipping")
        continue
    try:
        path = hf_hub_download(
            repo_id="MERChallenge/MER2026",
            repo_type="dataset",
            filename=f,
            token="$HF_TOKEN",
            local_dir="$FEATURES_DIR",
        )
        print(f"  Downloaded {f}")
    except Exception as e:
        print(f"  WARNING: Could not download {f}: {e}")
PYEOF

echo "=== Step 4: Update config.py ==="
sed -i "s|'MER2026Raw':   'xxx/dataset/mer2026-dataset'|'MER2026Raw':   '$DATA_ROOT/mer2026-dataset'|g" \
    /root/MERTools/MER2026/MER2026_Track1/config.py
sed -i "s|'MER2026':      'xxx/dataset/mer2026-dataset-process'|'MER2026':      '$FEATURES_DIR'|g" \
    /root/MERTools/MER2026/MER2026_Track1/config.py
sed -i "s|PATH_TO_PRETRAINED_MODELS = 'xxx/tools'|PATH_TO_PRETRAINED_MODELS = '$DATA_ROOT/tools'|g" \
    /root/MERTools/MER2026/MER2026_Track1/config.py

echo ""
echo "=== Verifying config.py ==="
grep -E "MER2026|PATH_TO_PRETRAINED" /root/MERTools/MER2026/MER2026_Track1/config.py

echo ""
echo "=== Step 5: Verify key files exist ==="
for f in \
    "$FEATURES_DIR/track1_label_6way.npz" \
    "$FEATURES_DIR/track1_track2_candidate.csv" \
    "$FEATURES_DIR/features/chinese-hubert-large-UTT" \
    "$FEATURES_DIR/features/chinese-macbert-large-UTT" \
    "$FEATURES_DIR/features/clip-vit-large-patch14-UTT"; do
    if [ -e "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done

echo ""
echo "=== Setup complete! ==="
echo "To run experiments:"
echo "  cd /root/MERTools/MER2026/MER2026_Track1"
echo "  bash run-cross-role.sh"
