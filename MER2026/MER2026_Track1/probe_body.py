"""
probe_body.py -- DE-RISK step for direction #2 (body pose + optical flow).

Before writing any pose-extraction / training code, answer ONE decisive
question: does the RAW video actually contain the listener's BODY, or is it a
tight head-and-shoulders / face crop?  Everything about #2 (est +2~3 WAF) hinges
on this. If clips are face-tight there is no body signal to extract and #2 is
dead on arrival -> we stop at 65.80.

Why a probe and not just "download and eyeball": this makes the answer both
QUANTITATIVE (face-box / frame-area coverage across a sample) and VISUAL (a
contact-sheet montage you can open). No GPU, no training, ~2 min, ~100 MB.

IMPORTANT -- must run ON THE GPU INSTANCE (or any box with open network).
HuggingFace is blocked by this dev box's egress policy, and the raw video lives
ONLY in the gated organizer repo `MERChallenge/MER2026` (the features repo
`hhieupt/mer2026-features` has NO raw video -- only WavLM/RoBERTa/CLIP feats).
You must have accepted the dataset terms on that repo; pass --hf_token if gated.

  pip install huggingface_hub opencv-python-headless numpy

  # 1) just see what raw-video assets the challenge repo ships (no download):
  python probe_body.py --list_only --hf_token $HF_TOKEN

  # 2) full probe: pull a small sample, measure coverage, write montage:
  python probe_body.py \
    --out_dir /workspace/body_probe \
    --n_clips 24 --frames_per_clip 3 \
    --hf_token $HF_TOKEN

Reads the verdict off stdout; open {out_dir}/montage_*.jpg to confirm by eye.

Coverage = median (face bbox area / full frame area) over sampled frames:
  > ~0.45  -> FACE-TIGHT crop        -> #2 INFEASIBLE (no body). Stop.
  ~0.15-0.45 -> head-and-shoulders   -> #2 PARTIAL (nod/lean/shoulder only).
  < ~0.15  -> wide / body visible    -> #2 VIABLE (arms/torso available).
Frames with NO detected face are also reported: many no-face frames on a
listener clip can itself indicate wide framing (face small/turned) -> body room.
"""
import os, sys, glob, argparse, io, zipfile, json

# repo ships a statistics.py that shadows stdlib `statistics`; strip repo dir
# from sys.path before importing cv2/torch-adjacent libs (see extract_fer.py).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _here]

import numpy as np
import cv2


VIDEO_EXTS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv')


def _human(n):
    for u in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or u == 'TB':
            return f'{n:.1f}{u}'
        n /= 1024


def list_assets(repo, token):
    """List repo files; return (all_files, video_like, zip_like) with sizes."""
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.repo_info(repo, repo_type='dataset', files_metadata=True, token=token)
    sizes = {}
    for s in info.siblings:
        sizes[s.rfilename] = s.size or 0
    files = sorted(sizes)
    vids = [f for f in files if f.lower().endswith(VIDEO_EXTS)]
    zips = [f for f in files if f.lower().endswith('.zip')]
    # heuristic: video-bearing zips usually have 'video' in the name and are big
    vzips = [f for f in zips if 'video' in f.lower()] or zips
    return files, sizes, vids, vzips


def sample_video_frames(path, k):
    """Return up to k frames (first/mid/last spread) as BGR arrays + (W,H)."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if n <= 0:
        # some containers don't report count; read sequentially
        ok, f = cap.read()
        while ok and len(frames) < k:
            frames.append(f)
            for _ in range(15):
                ok, f = cap.read()
        cap.release()
        wh = (frames[0].shape[1], frames[0].shape[0]) if frames else (0, 0)
        return frames, wh
    idxs = np.linspace(0, max(0, n - 1), k).astype(int)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    wh = (frames[0].shape[1], frames[0].shape[0]) if frames else (0, 0)
    return frames, wh


def face_coverage(frame, cascade):
    """Return (coverage_ratio_or_None, bbox_or_None). None => no face found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                     minSize=(24, 24))
    if len(faces) == 0:
        return None, None
    # largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    H, W = frame.shape[:2]
    return (w * h) / float(W * H), (x, y, w, h)


def make_montage(thumbs, path, cols=6, cell=160):
    if not thumbs:
        return
    rows = (len(thumbs) + cols - 1) // cols
    canvas = np.full((rows * cell, cols * cell, 3), 30, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        th = cv2.resize(t, (cell, cell))
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = th
    cv2.imwrite(path, canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='MERChallenge/MER2026')
    ap.add_argument('--out_dir', default='./body_probe')
    ap.add_argument('--n_clips', type=int, default=24)
    ap.add_argument('--frames_per_clip', type=int, default=3)
    ap.add_argument('--hf_token', default=os.environ.get('HF_TOKEN', '') or None)
    ap.add_argument('--list_only', action='store_true')
    ap.add_argument('--max_zip_mb', type=int, default=1500,
                    help='skip video zips larger than this when auto-picking')
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    files, sizes, vids, vzips = list_assets(args.repo, args.hf_token)
    print(f'\n=== {args.repo}: {len(files)} files ===')
    print(f'loose video files : {len(vids)}')
    print(f'zip archives      : {len([f for f in files if f.endswith(".zip")])} '
          f'(video-like: {len(vzips)})')
    for f in vzips[:12]:
        print(f'   {f:60s} {_human(sizes.get(f, 0))}')
    if vids[:8]:
        print('  sample loose videos:')
        for f in vids[:8]:
            print(f'   {f:60s} {_human(sizes.get(f, 0))}')
    if not vids and not vzips:
        print('\n!! No video assets found. #2 needs raw video the repo does not '
              'expose here -- check dataset access / other repo. STOP.')
        return
    if args.list_only:
        print('\n(list_only) -- rerun without --list_only to download & measure.')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    local_videos = []

    # Prefer loose video files (cheap: grab n_clips of them).
    if vids:
        pick = vids[:args.n_clips]
        print(f'\nDownloading {len(pick)} loose video files ...')
        for f in pick:
            p = hf_hub_download(args.repo, f, repo_type='dataset',
                                local_dir=args.out_dir, token=args.hf_token)
            local_videos.append(p)
    else:
        # Fall back to the smallest video-like zip, extract n_clips from it.
        vzips = sorted(vzips, key=lambda f: sizes.get(f, 1 << 60))
        cand = [f for f in vzips if sizes.get(f, 0) <= args.max_zip_mb * (1 << 20)]
        target = (cand or vzips)[0]
        print(f'\nNo loose videos; downloading smallest video zip: {target} '
              f'({_human(sizes.get(target, 0))}) ...')
        zp = hf_hub_download(args.repo, target, repo_type='dataset',
                             local_dir=args.out_dir, token=args.hf_token)
        with zipfile.ZipFile(zp) as z:
            members = [m for m in z.namelist()
                       if m.lower().endswith(VIDEO_EXTS)][:args.n_clips]
            for m in members:
                dst = os.path.join(args.out_dir, os.path.basename(m))
                with z.open(m) as src, open(dst, 'wb') as out:
                    out.write(src.read())
                local_videos.append(dst)

    if not local_videos:
        print('!! Downloaded assets but found no playable video files. STOP.')
        return

    # Measure coverage + collect thumbnails.
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    covs, sizes_wh, no_face, thumbs = [], [], 0, []
    for p in local_videos:
        frames, wh = sample_video_frames(p, args.frames_per_clip)
        if wh[0]:
            sizes_wh.append(wh)
        for fr in frames:
            cov, box = face_coverage(fr, cascade)
            vis = fr.copy()
            if box is not None:
                covs.append(cov)
                x, y, w, h = box
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
            else:
                no_face += 1
            thumbs.append(vis)

    montage = os.path.join(args.out_dir, 'montage_frames.jpg')
    make_montage(thumbs, montage)

    n_meas = len(covs)
    print('\n================= VERDICT =================')
    if sizes_wh:
        ws = np.array([w for w, h in sizes_wh]); hs = np.array([h for w, h in sizes_wh])
        print(f'frame size (median WxH): {int(np.median(ws))} x {int(np.median(hs))}')
    print(f'frames sampled           : {len(thumbs)}  (face found: {n_meas}, '
          f'no-face: {no_face})')
    if n_meas:
        med = float(np.median(covs))
        print(f'face coverage (median)   : {med:.3f}  '
              f'[min {min(covs):.3f}, max {max(covs):.3f}]')
        if med > 0.45:
            verdict = 'FACE-TIGHT crop -> #2 INFEASIBLE (no body). Recommend STOP at 65.80.'
        elif med > 0.15:
            verdict = ('HEAD-AND-SHOULDERS -> #2 PARTIAL (nod / lean / shoulder '
                       'tension only; no arms/torso). Modest upside.')
        else:
            verdict = 'WIDE framing -> #2 VIABLE (body/arms available). Proceed to build.'
    else:
        verdict = ('no faces detected in any sampled frame -- inspect montage by '
                   'eye; could be wide framing (good for #2) or a read error.')
        med = None
    print('=>', verdict)
    print(f'\nOpen the montage to confirm visually: {montage}')

    with open(os.path.join(args.out_dir, 'probe_summary.json'), 'w') as f:
        json.dump({'repo': args.repo, 'n_clips': len(local_videos),
                   'frames_sampled': len(thumbs), 'faces_found': n_meas,
                   'no_face': no_face,
                   'median_coverage': med,
                   'median_wh': [int(np.median([w for w, h in sizes_wh])),
                                 int(np.median([h for w, h in sizes_wh]))]
                                if sizes_wh else None,
                   'verdict': verdict}, f, indent=2)
    print('summary -> ' + os.path.join(args.out_dir, 'probe_summary.json'))


if __name__ == '__main__':
    main()
