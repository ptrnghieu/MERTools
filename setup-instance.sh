#!/bin/bash
# Setup script for new Vast.ai instance
# Usage: bash setup-instance.sh <HF_TOKEN> [DATA_ROOT]
# Example: bash setup-instance.sh hf_xxxx /root/data

set -e

HF_TOKEN="${1:?Usage: bash setup-instance.sh <HF_TOKEN> [DATA_ROOT]}"
DATA_ROOT="${2:-/root/data}"

FEATURES_DIR="$DATA_ROOT/mer2026-dataset-process"
mkdir -p "$FEATURES_DIR"

echo "=== Step 1: Install dependencies ==="
pip install -q fire huggingface_hub

echo ""
echo "=== Step 2: Clone MERTools repo ==="
if [ ! -d "/root/MERTools" ]; then
    git clone https://github.com/ptrnghieu/MERTools.git /root/MERTools
fi
cd /root/MERTools
git checkout claude/awesome-albattani-bpdlsk
git pull origin claude/awesome-albattani-bpdlsk

echo ""
echo "=== Step 3: Download features.zip from personal HF repo ==="
python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import os

path = hf_hub_download(
    repo_id="hhieupt/mer2026-features",
    repo_type="dataset",
    filename="features.zip",
    token="$HF_TOKEN",
    local_dir="$FEATURES_DIR",
)
print(f"Downloaded to: {path}")
PYEOF

echo ""
echo "=== Step 4: Extract features.zip ==="
cd "$FEATURES_DIR"
unzip -q features.zip
echo "Extracted. Contents:"
ls -lh "$FEATURES_DIR/"
rm features.zip
echo "Removed features.zip to save disk space"

echo ""
echo "=== Step 5: Download label + CSV files from MERChallenge/MER2026 ==="
python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import os

files = [
    "track1_train.csv",
    "track1_track2_candidate.csv",
    "track1_label_6way.npz",
    "track1_subtitle_chieng.csv",
]
for f in files:
    dest = os.path.join("$FEATURES_DIR", f)
    if os.path.exists(dest):
        print(f"  SKIP (exists): {f}")
        continue
    try:
        hf_hub_download(
            repo_id="MERChallenge/MER2026",
            repo_type="dataset",
            filename=f,
            token="$HF_TOKEN",
            local_dir="$FEATURES_DIR",
        )
        print(f"  OK: {f}")
    except Exception as e:
        print(f"  WARNING: {f} -> {e}")
PYEOF

echo ""
echo "=== Step 6: Update config.py ==="
CONFIG="/root/MERTools/MER2026/MER2026_Track1/config.py"
sed -i "s|'MER2026Raw':   'xxx/dataset/mer2026-dataset'|'MER2026Raw':   '$DATA_ROOT/mer2026-dataset'|g" "$CONFIG"
sed -i "s|'MER2026':      'xxx/dataset/mer2026-dataset-process'|'MER2026':      '$FEATURES_DIR'|g" "$CONFIG"
sed -i "s|PATH_TO_PRETRAINED_MODELS = 'xxx/tools'|PATH_TO_PRETRAINED_MODELS = '$DATA_ROOT/tools'|g" "$CONFIG"
echo "config.py updated:"
grep -E "MER2026|PRETRAINED" "$CONFIG"

echo ""
echo "=== Step 7: Verify key files ==="
ALL_OK=true
for f in \
    "$FEATURES_DIR/track1_label_6way.npz" \
    "$FEATURES_DIR/track1_track2_candidate.csv" \
    "$FEATURES_DIR/features/chinese-hubert-large-UTT" \
    "$FEATURES_DIR/features/chinese-macbert-large-UTT" \
    "$FEATURES_DIR/features/clip-vit-large-patch14-UTT"; do
    if [ -e "$f" ]; then
        echo "  OK: $(basename $f)"
    else
        echo "  MISSING: $f"
        ALL_OK=false
    fi
done

echo ""
if [ "$ALL_OK" = true ]; then
    echo "=== Setup complete! Ready to train. ==="
    echo ""
    echo "Run experiments:"
    echo "  cd /root/MERTools/MER2026/MER2026_Track1"
    echo "  tmux new -s train"
    echo "  bash run-cross-role.sh"
else
    echo "=== Setup done with warnings. Check MISSING files above. ==="
fi
