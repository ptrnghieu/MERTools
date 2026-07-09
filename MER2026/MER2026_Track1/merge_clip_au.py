"""
Tang 1 (zero code risk): concatenate au_dynamics features onto CLIP video
features per frame -> clip_au-FRA/{name}.npy. Train v3 unchanged with
--video_feature=clip_au-FRA (the video encoder auto-adapts to the new dim).

AU sequence is resampled to CLIP's frame count per sample. Samples missing an
AU file get zero-padded AU columns so every output has the SAME feature dim
(required by the encoder).

Usage (on the training instance, after downloading au_dynamics-FRA from HF):
  python3 merge_clip_au.py \
    --clip_dir /workspace/mer2026/embeddings/clip-vit-large-patch14-FRA \
    --au_dir   /workspace/mer2026/embeddings/au_dynamics-FRA \
    --out_dir  /workspace/mer2026/embeddings/clip_au-FRA
"""
import os, glob, argparse
import numpy as np


def resample(x, T):
    if x.shape[0] == T:
        return x
    idx = np.linspace(0, x.shape[0] - 1, T).astype(int)
    return x[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_dir', required=True)
    ap.add_argument('--au_dir',   required=True)
    ap.add_argument('--out_dir',  required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # infer AU dim from any available au file
    au_files = glob.glob(os.path.join(args.au_dir, '*.npy'))
    assert au_files, f'no AU files in {args.au_dir}'
    au_dim = np.load(au_files[0]).shape[1]
    print(f'AU dim = {au_dim}; {len(au_files)} au files present')

    miss = 0
    clip_files = glob.glob(os.path.join(args.clip_dir, '*.npy'))
    for f in clip_files:
        name = os.path.basename(f)
        clip = np.load(f).astype(np.float32)          # (Tc, 768)
        aup = os.path.join(args.au_dir, name)
        if os.path.exists(aup):
            au = resample(np.load(aup).astype(np.float32), clip.shape[0])
        else:
            au = np.zeros((clip.shape[0], au_dim), np.float32)
            miss += 1
        out = np.concatenate([clip, au], axis=1)      # (Tc, 768+au_dim)
        np.save(os.path.join(args.out_dir, name), out)
    print(f'DONE: {len(clip_files)} merged, {miss} missing-AU (zero-padded) -> {args.out_dir}')


if __name__ == '__main__':
    main()
