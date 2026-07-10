"""
Build clip_au20-FRA = [CLIP video features | AU-only (20 FACS intensities)]
for memocmt_v9, which splits it into a CLIP branch + an AU branch.

Design choices (from transfer analysis):
  - AU-ONLY: keep the first 20 dims (AU intensities); DROP the 136 landmark +
    6 head-pose dims. Raw 2D landmarks / pose encode head position, face shape,
    identity and head-pose -- and head-pose is the strongest talking->listening
    shift, i.e. the WORST-transferring signal. AU intensities are normalized,
    pose/identity-invariant muscle activations -> the cleanest transfer.
  - Keep AU at its own frame count and UPSAMPLE CLIP to it (not the reverse),
    so AU's temporal resolution (micro-expression dynamics) is preserved; the
    model gives AU its own LSTM branch.
  - Interpolate all-zero (detection-fail) AU rows from neighbours.
  - Per-AU z-score using TRAIN-only stats (unsupervised, no label leak).

Usage:
  python3 merge_clip_au.py \
    --clip_dir embeddings/clip-vit-large-patch14-FRA \
    --au_dir   embeddings/au_dynamics-FRA \
    --out_dir  embeddings/clip_au20-FRA \
    --label_npz /workspace/mer2026/track1_label_6way.npz
"""
import os, glob, argparse
import numpy as np

AU_N = 20   # first 20 dims of au_dynamics = FACS AU intensities


def resample(x, T):
    if x.shape[0] == T:
        return x
    idx = np.linspace(0, x.shape[0] - 1, T).astype(int)
    return x[idx]


def interp_zero_rows(a):
    bad = ~np.any(a != 0, axis=1)
    if not bad.any() or bad.all():
        return a
    good = np.where(~bad)[0]
    for i in np.where(bad)[0]:
        a[i] = a[good[np.argmin(np.abs(good - i))]]
    return a


def load_au20(path):
    a = np.load(path).astype(np.float64)          # (T, 162)
    a = interp_zero_rows(a)
    return a[:, :AU_N]                             # (T, 20) AU intensities only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip_dir', required=True)
    ap.add_argument('--au_dir',   required=True)
    ap.add_argument('--out_dir',  required=True)
    ap.add_argument('--label_npz', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # train-only z-score stats over the 20 AU dims
    train_names = set(np.load(args.label_npz, allow_pickle=True)['train_corpus'].tolist().keys())
    n, s, ss = 0, np.zeros(AU_N), np.zeros(AU_N)
    for name in train_names:
        p = os.path.join(args.au_dir, name + '.npy')
        if not os.path.exists(p):
            continue
        a = load_au20(p)
        n += a.shape[0]; s += a.sum(0); ss += (a * a).sum(0)
    mean = s / max(n, 1)
    std = np.sqrt(np.maximum(ss / max(n, 1) - mean ** 2, 0)) + 1e-6
    print(f'AU z-score from {n} train frames | mean={mean.round(2)}')

    miss = 0
    clip_files = glob.glob(os.path.join(args.clip_dir, '*.npy'))
    for f in clip_files:
        name = os.path.basename(f)
        clip = np.load(f).astype(np.float32)               # (Tc, 768)
        aup = os.path.join(args.au_dir, name)
        if os.path.exists(aup):
            au = (load_au20(aup) - mean) / std             # (Ta, 20) z-scored
            au = au.astype(np.float32)
        else:
            au = np.zeros((clip.shape[0], AU_N), np.float32)
            miss += 1
        clip_rs = resample(clip, au.shape[0])              # upsample CLIP to AU length
        out = np.concatenate([clip_rs, au], axis=1)        # (Ta, 768+20)
        np.save(os.path.join(args.out_dir, name), out)
    print(f'DONE: {len(clip_files)} merged, {miss} missing-AU (zero) -> {args.out_dir} '
          f'(dim {768 + AU_N})')


if __name__ == '__main__':
    main()
