#!/bin/bash
# Sequential multi-seed x multi-arch x multi-hyperparam ensemble generator.
# Trains only the STRONG (~65-66) architectures (v3, v6, v11); the <64 variants
# (v2/v4/v5/v7/v8/v9/v10) are excluded -- they would drag the ensemble down.
# Runs SEQUENTIALLY in the foreground (never background/Ctrl+Z -> OOM). Resumable:
# a config whose npz already exists is skipped, so re-running continues.
#
# Usage:
#   cd /workspace/MERTools/MER2026/MER2026_Track1
#   tmux new -s ens
#   bash run_ensemble.sh 2>&1 | tee ensemble.log
# Then gather + submit (see bottom of this file).
set -u
cd "$(dirname "$0")"
mkdir -p ensemble_npz

COMMON="--optimizer=adamw --l2=0.00001 --use_lr_scheduler --epochs=100 --patience=10 \
--label_smoothing=0.0 --gpu=0 --use_class_weight --dataset=MER2026 --feat_type=frm_unalign \
--audio_feature=wavlm-large-FRA --text_feature=chinese-roberta-wwm-ext-large-FRA \
--video_feature=clip-vit-large-patch14-FRA"

run() {  # $1=model  $2=hyper_yaml  $3=seed  $4=tag
  if [ -f "ensemble_npz/$4.npz" ]; then echo "SKIP $4 (exists)"; return; fi
  echo "===== TRAIN $4 (model=$1 hyper=$2 seed=$3) ====="
  python3 -u main-release.py --model="$1" --hyper_path="toolkit/hyper-sweep/$2" --seed="$3" $COMMON || { echo "FAILED $4"; return; }
  npz="$(ls -t ./saved-trimodal/result/test1_*model:$1*.npz | head -1)"
  cp "$npz" "ensemble_npz/$4.npz" && echo "SAVED ensemble_npz/$4.npz"
}

# v3 (best arch) -- multi-seed
run memocmt_v3  v3_baseline.yaml 42 v3_s42
run memocmt_v3  v3_baseline.yaml 0  v3_s0
run memocmt_v3  v3_baseline.yaml 1  v3_s1
run memocmt_v3  v3_baseline.yaml 2  v3_s2
run memocmt_v3  v3_baseline.yaml 3  v3_s3
# v3 -- hyperparam diversity
run memocmt_v3  v3_h256.yaml     42 v3_h256_s42
run memocmt_v3  v3_sdp05.yaml    42 v3_sdp05_s42
# v6 (CM-StEW) -- multi-seed
run memocmt_v6  v6_baseline.yaml 42 v6_s42
run memocmt_v6  v6_baseline.yaml 0  v6_s0
run memocmt_v6  v6_baseline.yaml 1  v6_s1
# v11 (deep-sup video) -- multi-seed
run memocmt_v11 v11_v05.yaml     42 v11_s42
run memocmt_v11 v11_v05.yaml     0  v11_s0
run memocmt_v11 v11_v05.yaml     1  v11_s1

echo "===== ALL DONE ====="
ls -la ensemble_npz/

# ---- Gather + submit (run manually after training) ----
# NPZS=$(ls ensemble_npz/*.npz | paste -sd,)
# for T in 1.5 2.0 2.5; do echo "== tau=$T =="; \
#   python3 submission.py ensemble_submission --result_npzs="$NPZS" --save_csv="answer.csv" --tau=$T; done
# rm -f answer.zip && zip -q answer.zip answer.csv     # submit best tau
