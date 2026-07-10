"""
Build clip_au-FRA = [CLIP video features | prepped AU/landmark/pose] per frame,
used by memocmt_v9 (which splits it back into a CLIP branch + an AU branch).

AU prep (addresses train/test confounds):
  (1) interpolate all-zero AU rows (OpenFace/py-feat detection failures that
      extract_au wrote as nan->0) from neighbouring frames;
  (2) per-column z-score using TRAIN-only statistics (unsupervised, no label
      leak) so raw AU intensities / landmarks / pose share a scale.

Then resample AU to CLIP's frame count and concat.

Usage:
  python3 merge_clip_au.py \
    --clip_dir  embeddings/clip-vit-large-patch14-FRA \
    --au_dir    embeddings/au_dynamics-FRA \
    --out_dir   embeddings/clip_au-FRA \
    --label_npz /workspace/mer2026/track1_label_6way.npz
"""
import os, glob, argparse
import numpy as np


def resample(x, T):
    if x.shape[0] == T:
        return x
    idx = np.linspace(0, x.shape[0] - 1, T).astype(int)
    return x[idx]


def interp_zero_rows(a):
    """Linearly interpolate rows that are all-zero (detection failures)."""
    bad = ~np.any(a != 0, axis=1)          # True where row is all zeros
    if not bad.any() or bad.all():
        return a
    good = np.where(~bad)[0]
    for i in np.where(bad)[0]:
        j = good[np.argmin(np.abs(good - i))]   # nearest good frame
        a[i] = a[j]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_dir', required=True)
    ap.add_argument('--au_dir',   required=True)
    ap.add_argument('--out_dir',  required=True)
    ap.add_argument('--label_npz', required=True, help='track1_label_6way.npz for train names -> z-score stats')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    au_files = glob.glob(os.path.join(args.au_dir, '*.npy'))
    assert au_files, f'no AU files in {args.au_dir}'
    au_dim = np.load(au_files[0]).shape[1]
    print(f'AU dim = {au_dim}; {len(au_files)} au files present')

    # --- train-only z-score stats (mean/std per column over train frames) ---
    train_names = set(np.load(args.label_npz, allow_pickle=True)['train_corpus'].tolist().keys())
    n, s, ss = 0, np.zeros(au_dim), np.zeros(au_dim)
    for name in train_names:
        p = os.path.join(args.au_dir, name + '.npy')
        if not os.path.exists(p):
            continue
        a = interp_zero_rows(np.load(p).astype(np.float64))
        n += a.shape[0]; s += a.sum(0); ss += (a * a).sum(0)
    mean = s / max(n, 1)
    std = np.sqrt(np.maximum(ss / max(n, 1) - mean ** 2, 0)) + 1e-6
    print(f'z-score stats from {n} train frames | mean[:3]={mean[:3].round(3)} std[:3]={std[:3].round(3)}')

    # --- build clip_au ---
    miss = 0
    clip_files = glob.glob(os.path.join(args.clip_dir, '*.npy'))
    for f in clip_files:
        name = os.path.basename(f)
        clip = np.load(f).astype(np.float32)               # (Tc, 768)
        aup = os.path.join(args.au_dir, name)
        if os.path.exists(aup):
            au = interp_zero_rows(np.load(aup).astype(np.float64))
            au = (au - mean) / std                          # z-score (train stats)
            au = resample(au.astype(np.float32), clip.shape[0])
        else:
            au = np.zeros((clip.shape[0], au_dim), np.float32)
            miss += 1
        out = np.concatenate([clip, au], axis=1)            # (Tc, 768+au_dim)
        np.save(os.path.join(args.out_dir, name), out)
    print(f'DONE: {len(clip_files)} merged, {miss} missing-AU (zero) -> {args.out_dir}')


if __name__ == '__main__':
    main()
