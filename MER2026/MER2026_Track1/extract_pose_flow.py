"""
extract_pose_flow.py -- OPTION 2: listener body-language features (pose + flow).

CLIP-ViT reads only the cropped FACE. In dyadic conversation the listener (s2)
also reacts with body language -- nodding/shaking the head, shrugging, hand
gestures of agreement/disagreement. This script extracts, per sampled frame:

  * POSE (MediaPipe Pose): 9 upper-body keypoints (nose, both shoulders, elbows,
    wrists, hips). Each -> (x, y, visibility). x,y are normalized by the
    shoulder-center and shoulder-width so the descriptor is translation/scale
    invariant.                                             -> 27 dims
  * OPTICAL FLOW (Farneback) between consecutive sampled frames: global mean/std
    magnitude, mean (dx, dy), a 2x2 spatial grid of mean magnitude (head vs
    hands region), and an 8-bin magnitude-weighted direction histogram.  -> 16 dims

Per-frame vector = 43 dims. Output matches the toolkit format exactly: one npy
per sample at {out_dir}/{name}.npy, shape (n_frames, 43). main-release.py reads
it as a --video_feature <dir> with --feat_type frm_unalign (loader avg-pools /12).

Runs on CPU (MediaPipe + Farneback); no GPU needed -- so it can run on the
extraction box while a GPU trains. Same iter/stream/upload/resume plumbing as
extract_clip_hires.py, and the SAME name scheme, so train + candidate splits go
into the SAME out_dir without collision, then zip+upload to HF.

Env:
  pip install mediapipe opencv-python-headless huggingface_hub numpy tqdm

SMOKE first (20 clips), then full:
  python extract_pose_flow.py \
    --repo MERChallenge/MER2026 \
    --asset video_7z/video_track1_train/video_split.zip \
    --out_dir /workspace/feats/mediapipe-pose-flow-FRA \
    --limit 20                       # drop --limit for the full run

  # candidate splits (same out_dir):
  python extract_pose_flow.py --repo MERChallenge/MER2026 \
    --asset video_7z/video_track1_track2_candidate/video_split_0001.zip \
    --out_dir /workspace/feats/mediapipe-pose-flow-FRA
  python extract_pose_flow.py --repo MERChallenge/MER2026 \
    --asset video_7z/video_track1_track2_candidate/video_split_0002.zip \
    --out_dir /workspace/feats/mediapipe-pose-flow-FRA

  # persist to HF:
  python extract_pose_flow.py --out_dir /workspace/feats/mediapipe-pose-flow-FRA \
    --upload_repo hhieupt/mer2026-features
"""
import os, sys, time, argparse, zipfile, shutil

# repo ships a statistics.py that shadows stdlib `statistics`; strip repo dir
# from sys.path before importing cv2/mediapipe (see extract_clip_hires.py).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _here]

import numpy as np
import cv2

VIDEO_EXTS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv')
SHM = '/dev/shm' if os.path.isdir('/dev/shm') else None

# MediaPipe Pose landmark indices (upper body)
POSE_SEL = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # nose, L/R shoulder, elbow, wrist, hip
POSE_DIM = len(POSE_SEL) * 3      # (x, y, visibility) per keypoint = 27
FLOW_DIM = 4 + 4 + 8             # mean/std mag + mean dx/dy + 2x2 grid + 8-bin dir = 16
FEAT_DIM = POSE_DIM + FLOW_DIM   # 43
FLOW_SIZE = 128                  # resize gray frames to this for Farneback (speed)


def uniform_idx(vlen, n):
    if vlen <= 0:
        return []
    if vlen <= n:
        return list(range(vlen))
    return np.linspace(0, vlen - 1, n).astype(int).tolist()


def read_frames(video_bytes, max_frames, name):
    """Decode up to max_frames uniformly-sampled RGB frames from raw bytes."""
    tmp = os.path.join(SHM or '.', f'_pf_{os.getpid()}_{name}') + '.mp4'
    with open(tmp, 'wb') as f:
        f.write(video_bytes)
    try:
        cap = cv2.VideoCapture(tmp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = uniform_idx(n, max_frames) if n > 0 else None
        frames = []
        if idxs is not None:
            want = set(idxs)
            i = 0
            ok, fr = cap.read()
            while ok:
                if i in want:
                    frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                i += 1
                ok, fr = cap.read()
        else:
            ok, fr = cap.read()
            while ok and len(frames) < max_frames:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                for _ in range(2):
                    ok, fr = cap.read()
        cap.release()
        return frames
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def pose_feat(landmarks):
    """9 upper-body keypoints -> (x, y, visibility), x/y normalized by
    shoulder-center + shoulder-width. Returns zeros if landmarks is None."""
    if landmarks is None:
        return np.zeros(POSE_DIM, dtype=np.float32)
    ls = np.array([landmarks[11].x, landmarks[11].y], dtype=np.float32)
    rs = np.array([landmarks[12].x, landmarks[12].y], dtype=np.float32)
    center = (ls + rs) / 2.0
    sw = float(np.linalg.norm(ls - rs)) + 1e-6
    out = np.zeros((len(POSE_SEL), 3), dtype=np.float32)
    for j, idx in enumerate(POSE_SEL):
        lm = landmarks[idx]
        out[j, 0] = (lm.x - center[0]) / sw
        out[j, 1] = (lm.y - center[1]) / sw
        out[j, 2] = lm.visibility
    return out.flatten()


def flow_feat(prev_gray, cur_gray):
    """Dense Farneback optical flow -> 16-dim global motion descriptor."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)
    feats = [float(mag.mean()), float(mag.std()), float(fx.mean()), float(fy.mean())]
    h, w = mag.shape
    for gy in range(2):
        for gx in range(2):
            feats.append(float(mag[gy*h//2:(gy+1)*h//2, gx*w//2:(gx+1)*w//2].mean()))
    hist = np.zeros(8, dtype=np.float32)
    bins = (ang / (2 * np.pi) * 8).astype(int) % 8
    for b in range(8):
        hist[b] = float(mag[bins == b].sum())
    hist /= (hist.sum() + 1e-6)
    feats.extend(hist.tolist())
    return np.array(feats, dtype=np.float32)


def extract_sample(frames, pose):
    """frames: list of RGB. Returns (T, FEAT_DIM) float32."""
    grays = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (FLOW_SIZE, FLOW_SIZE))
             for f in frames]
    feats = []
    prev_gray = None
    for t, frame in enumerate(frames):
        res = pose.process(frame)                    # frame is RGB (MediaPipe wants RGB)
        lm = res.pose_landmarks.landmark if res.pose_landmarks else None
        pf = pose_feat(lm)
        if prev_gray is None:
            ff = np.zeros(FLOW_DIM, dtype=np.float32)  # first frame: no flow
        else:
            ff = flow_feat(prev_gray, grays[t])
        prev_gray = grays[t]
        feats.append(np.concatenate([pf, ff]))
    return np.stack(feats, 0).astype(np.float32)


def iter_members(args):
    """Yield (name, bytes) for each video in the asset (remote-range or local)."""
    from huggingface_hub import HfFileSystem, hf_hub_download
    if args.stream:
        fs = HfFileSystem(token=args.hf_token)
        rpath = f'datasets/{args.repo}/{args.asset}'
        f = fs.open(rpath, 'rb')
        zf = zipfile.ZipFile(f)
        members = [m for m in zf.namelist() if m.lower().endswith(VIDEO_EXTS)]
        members.sort()
        for m in members:
            yield os.path.splitext(os.path.basename(m))[0], zf.read(m)
        zf.close(); f.close()
    else:
        lp = hf_hub_download(args.repo, args.asset, repo_type='dataset',
                             local_dir=args.dl_dir, token=args.hf_token)
        with zipfile.ZipFile(lp) as zf:
            members = [m for m in zf.namelist()
                       if m.lower().endswith(VIDEO_EXTS)]
            members.sort()
            for m in members:
                yield os.path.splitext(os.path.basename(m))[0], zf.read(m)


def do_upload(local_dir, repo, token):
    from huggingface_hub import HfApi
    local_dir = local_dir.rstrip('/')
    parent = os.path.dirname(os.path.abspath(local_dir))
    base = os.path.basename(local_dir)
    zip_base = os.path.join(parent, base)
    print(f'zipping {local_dir} -> {zip_base}.zip ...')
    shutil.make_archive(zip_base, 'zip', root_dir=parent, base_dir=base)
    print(f'uploading {base}.zip to {repo} ...')
    HfApi().upload_file(path_or_fileobj=zip_base + '.zip',
                        path_in_repo=base + '.zip',
                        repo_id=repo, repo_type='dataset', token=token)
    print('upload done.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='MERChallenge/MER2026')
    ap.add_argument('--asset', default='', help='zip path inside the repo')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_frames', type=int, default=64)
    ap.add_argument('--model_complexity', type=int, default=1,
                    help='MediaPipe Pose complexity 0/1/2 (higher=slower/accurate)')
    ap.add_argument('--min_det_conf', type=float, default=0.5)
    ap.add_argument('--stream', action='store_true',
                    help='read clips from the remote zip via range requests')
    ap.add_argument('--dl_dir', default='/workspace/_zip_dl')
    ap.add_argument('--limit', type=int, default=0, help='smoke: first N clips')
    ap.add_argument('--hf_token', default=os.environ.get('HF_TOKEN') or None)
    ap.add_argument('--upload_repo', default='')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.upload_repo and not args.asset:
        do_upload(args.out_dir, args.upload_repo, args.hf_token)
        return

    import mediapipe as mp
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,                 # sampled frames are non-consecutive
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_det_conf,
    )
    print(f'MediaPipe Pose ready (complexity={args.model_complexity}); '
          f'feat_dim={FEAT_DIM} (pose {POSE_DIM} + flow {FLOW_DIM})')

    done = set(os.path.splitext(f)[0] for f in os.listdir(args.out_dir)
               if f.endswith('.npy'))
    print(f'resuming: {len(done)} already extracted')

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    for i, (name, vb) in enumerate(iter_members(args)):
        if args.limit and i >= args.limit:
            break
        if name in done:
            n_skip += 1
            continue
        try:
            frames = read_frames(vb, args.max_frames, name)
            if not frames:
                raise RuntimeError('no frames decoded')
            feat = extract_sample(frames, pose)         # (T, 43)
            np.save(os.path.join(args.out_dir, name + '.npy'), feat)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f'  !! {name}: {type(e).__name__} {str(e)[:120]}')
            continue
        if (i + 1) % 200 == 0:
            r = (time.time() - t0) / max(1, n_ok)
            print(f'[{i + 1}] ok={n_ok} skip={n_skip} err={n_err} '
                  f'{r*1000:.0f}ms/clip', flush=True)

    pose.close()
    print(f'\nDONE: ok={n_ok} skip={n_skip} err={n_err} '
          f'in {(time.time()-t0)/60:.1f} min -> {args.out_dir}')
    if args.upload_repo:
        do_upload(args.out_dir, args.upload_repo, args.hf_token)


if __name__ == '__main__':
    main()
