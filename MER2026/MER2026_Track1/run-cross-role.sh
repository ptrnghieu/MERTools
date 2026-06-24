#!/bin/bash
# Cross-role transferability experiments
# Run trimodal with cross_role_attention and compare against attention baseline

for ((i=1; i<=50; i++)); do
    echo "=== Run $i ==="
    python -u main-release.py \
        --model=cross_role_attention \
        --feat_type='utt' \
        --dataset=MER2026 \
        --audio_feature='chinese-hubert-large-UTT' \
        --text_feature='chinese-macbert-large-UTT' \
        --video_feature='clip-vit-large-patch14-UTT' \
        --gpu=0
done

echo "Done!"
