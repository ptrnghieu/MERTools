"""
Extract AU + landmark + head-pose dynamics from OpenFace face crops using
py-feat (Detectorv1), producing an identity-invariant, emotion-relevant
per-frame feature sequence per sample -> au_dynamics-FRA/{name}.npy (T, D).

Run from a NON-repo dir (repo has a statistics.py that shadows stdlib and
breaks seaborn/py-feat). Resumable (skips existing outputs) and can push to
HF periodically so an ephemeral instance reclaim does not lose work.

Usage (probe first!):
  cd /workspace/mer2026
  python3 /workspace/MERTools/MER2026/MER2026_Track1/extract_au.py \
     --face_root ./raw_face/openface_face --out_dir ./au_dynamics-FRA \
     --n_frms 10 --limit 3            # probe: prints resolved dims + AU cols
  # then full run + upload:
  python3 .../extract_au.py --face_root ./raw_face/openface_face \
     --out_dir ./au_dynamics-FRA --n_frms 10 \
     --upload_repo hhieupt/mer2026-features --upload_every 2000
"""
import sys, os
# The repo dir contains a statistics.py that shadows Python's stdlib and breaks
# seaborn -> py-feat. Running this script by its repo path puts that dir on
# sys.path[0]; strip it so stdlib `statistics` wins.
_self = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != _self]

import glob, argparse, tempfile
import numpy as np


def uniform_idx(vlen, n):
    if vlen <= 0:
        return [0] * n
    if vlen <= n:
        return list(range(vlen)) + [vlen - 1] * (n - vlen)
    return np.linspace(0, vlen - 1, n).astype(int).tolist()


def _fit_rows(arr, n):
    """Force arr to exactly n rows (pad by repeat / subsample)."""
    if arr.shape[0] == n:
        return arr
    idx = uniform_idx(arr.shape[0], n)
    return arr[idx]


def _upload(local_dir, repo):
    from huggingface_hub import upload_folder
    base = os.path.basename(local_dir.rstrip('/'))
    upload_folder(repo_id=repo, repo_type='dataset',
                  folder_path=local_dir, path_in_repo=base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--face_root', required=True, help='dir with {name}/{name}.npy crops')
    ap.add_argument('--out_dir',   required=True, help='output feature dir (au_dynamics-FRA)')
    ap.add_argument('--n_frms',    type=int, default=10)
    ap.add_argument('--device',    default='cuda')
    ap.add_argument('--limit',     type=int, default=0, help='probe on first N samples (0=all)')
    ap.add_argument('--upload_repo',  default='')
    ap.add_argument('--upload_every', type=int, default=2000)
    args = ap.parse_args()

    from feat.detector import Detectorv1 as Detector
    import imageio.v2 as imageio

    det = Detector(device=args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    dirs = sorted(glob.glob(os.path.join(args.face_root, '*')))
    if args.limit:
        dirs = dirs[:args.limit]
    tmp = tempfile.mkdtemp()
    done, skipped, printed = 0, 0, False

    for d in dirs:
        name = os.path.basename(d)
        out = os.path.join(args.out_dir, name + '.npy')
        if os.path.exists(out):
            skipped += 1
            continue
        npys = [x for x in glob.glob(os.path.join(d, '*')) if x.endswith('.npy')]
        if not npys:
            continue
        fr = np.load(npys[0])                       # (T, H, W, 3) uint8
        idx = uniform_idx(len(fr), args.n_frms)
        paths = []
        for k, ii in enumerate(idx):
            p = os.path.join(tmp, f'{k}.png')
            imageio.imwrite(p, fr[ii])
            paths.append(p)

        try:
            fex = det.detect(paths)
        except TypeError:
            fex = det.detect(paths, data_type='image')

        au = np.nan_to_num(fex.aus.to_numpy().astype(np.float32))
        po = np.nan_to_num(fex.poses.to_numpy().astype(np.float32))
        lm = np.nan_to_num(fex.landmarks.to_numpy().astype(np.float32))
        # per-frame standardize landmarks -> scale/position invariant
        lm = (lm - lm.mean(axis=1, keepdims=True)) / (lm.std(axis=1, keepdims=True) + 1e-6)

        feat = np.concatenate([au, lm, po], axis=1)  # (rows, D)
        feat = _fit_rows(feat, args.n_frms).astype(np.float32)
        np.save(out, feat)

        if not printed:
            print(f'[dims] au={au.shape[1]} lm={lm.shape[1]} pose={po.shape[1]} '
                  f'-> D={feat.shape[1]} | seq_len={feat.shape[0]}')
            print('[au cols]', list(fex.aus.columns))
            print('[pose cols]', list(fex.poses.columns))
            printed = True
        done += 1
        if args.upload_repo and done % args.upload_every == 0:
            _upload(args.out_dir, args.upload_repo)
            print(f'... {done} extracted, pushed to HF')

    if args.upload_repo and done:
        _upload(args.out_dir, args.upload_repo)
    print(f'DONE: {done} new, {skipped} skipped -> {args.out_dir}')


if __name__ == '__main__':
    main()
