#!/bin/bash
# BBA-style Target-related Model Adaptation (Tier-1) for MER-Cross.
# Runs INDEPENDENTLY of run_ensemble.sh -- separate script, separate outputs
# under ./tma_out, does NOT touch main-release.py / memocmt_v3.py, so it is
# safe to run in a SECOND tmux window alongside a running ensemble (share the
# GPU only if memory allows; otherwise run after the ensemble finishes).
#
#   cd /workspace/MERTools/MER2026/MER2026_Track1
#   tmux new -s tma
#   bash run_tma.sh 2>&1 | tee tma.log
#
# config.py must point PATH_TO_LABEL['MER2026'] at the REAL track1_label_6way.npz
# (same requirement as the ensemble). Override per-run with --label_npz.
set -u
cd "$(dirname "$0")"
mkdir -p tma_out

# ---- Step 1: BRIDGE -- train one v3, save weights (frozen speaker branch) +
#      test1 logits over the candidate pool (pseudo-label source). ~40 min. ----
python3 -u tma_finetune.py --mode bridge \
  --bridge_ckpt ./tma_out/bridge_v3.pt \
  --bridge_logits ./tma_out/bridge_v3_test1.npz

# OPTIONAL: for stronger pseudo-labels, reuse your best 5-fold v3 npz instead of
# the single-fold bridge logits -- just point --bridge_logits at it in Step 2.

# ---- Step 2: TMA -- re-init {listener_pool, fusion, fc_out}, freeze speaker +
#      video encoders, adapt ONLY the head on the candidate pseudo pool. ----
python3 -u tma_finetune.py --mode tma --tag t1 \
  --bridge_ckpt ./tma_out/bridge_v3.pt \
  --bridge_logits ./tma_out/bridge_v3_test1.npz \
  --tma_lr 1e-4 --tma_epochs 20 --tma_speaker_drop 0.3 --tau 2.5 --top_frac 0.5 \
  --out_npz ./tma_out/test1_tma_t1.npz

echo "===== TMA DONE ====="
ls -la tma_out/

# ---- Step 3: tau-sweep + submit (compare against v3's 65.78) ----
# for T in 2.0 2.5 3.0; do echo "== tau=$T =="; \
#   python3 submission.py adjust_submission --result_npz=./tma_out/test1_tma_t1.npz \
#     --save_csv=answer.csv --tau=$T; done
# rm -f answer.zip && zip -q answer.zip answer.csv     # submit best tau
#
# It is also a decorrelated ENSEMBLE member: add test1_tma_t1.npz to the
# ensemble_submission --result_npzs list once you confirm it is >= ~65.
