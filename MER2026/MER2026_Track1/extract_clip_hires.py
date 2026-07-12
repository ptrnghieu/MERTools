"""
extract_clip_hires.py -- OPTION 1: high-resolution video features.

Attacks the #1 diagnosed bottleneck (weak visual perception): v3 (65.78) reads
CLIP features extracted from OpenFace 112x112 crops that were UPSAMPLED to CLIP's
224 input -- detail the network never had. The raw MERChallenge/MER2026 video is
720px scene footage; the face within it is ~150-250px tall. Re-detect the face on
the FULL frame, crop it at native resolution, and feed CLIP-ViT-L/14-336 at its
native 336 input. Same 768-dim output, same architecture -- ONLY the input image
quality changes, so retraining v3 on these isolates that one variable.

Output matches the toolkit format exactly: one npy per sample at
{out_dir}/{name}.npy, shape (n_frames, 768). main-release.py then reads it as
--video_feature <dir> with --feat_type frm_unalign (loader avg-pools /12).

Run the SAME way for train and every candidate split into the SAME out_dir
(names don't collide). Then zip+upload out_dir to HF (instance is ephemeral) and
retrain v3 swapping only --video_feature.

Env (A100):
  pip install transformers facenet-pytorch opencv-python-headless huggingface_hub tqdm pillow

SMOKE first (20 clips), then full:
  python extract_clip_hires.py \
    --repo MERChallenge/MER2026 \
    --asset video_7z/video_track1_train/video_split.zip \
    --out_dir /workspace/feats/clip-vit-large-patch14-336-hires-FRA \
    --limit 20                       # drop --limit for the full run

  # candidate (two splits, same out_dir):
  python extract_clip_hires.py --repo MERChallenge/MER2026 \
    --asset video_7z/video_track1_track2_candidate/video_split_0001.zip \
    --out_dir /workspace/feats/clip-vit-large-patch14-336-hires-FRA
  python extract_clip_hires.py --repo MERChallenge/MER2026 \
    --asset video_7z/video_track1_track2_candidate/video_split_0002.zip \
    --out_dir /workspace/feats/clip-vit-large-patch14-336-hires-FRA

  # persist to HF:
  python extract_clip_hires.py --out_dir /workspace/feats/clip-...-FRA \
    --upload_repo hhieupt/mer2026-features
"""
import os, sys, glob, time, argparse, zipfile, io, shutil

# repo ships a statistics.py that shadows stdlib `statistics`; strip repo dir
# from sys.path before importing torch-adjacent libs (see extract_fer.py).
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != _here]

import numpy as np
import cv2
from PIL import Image

VIDEO_EXTS = ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv')
SHM = '/dev/shm' if os.path.isdir('/dev/shm') else None


def uniform_idx(vlen, n):
    if vlen <= 0:
        return []
    if vlen <= n:
        return list(range(vlen))
    return np.linspace(0, vlen - 1, n).astype(int).tolist()


def read_frames(video_bytes, max_frames, name):
    """Decode up to max_frames uniformly-sampled RGB frames from raw bytes."""
    tmp = os.path.join(SHM or '.', f'_clip_{os.getpid()}_{name}')
    tmp += '.mp4'
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
        else:  # unknown count: read sequentially, cap at max_frames*stride
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


def crop_faces(frames, mtcnn, margin, out_size):
    """Detect the largest face per frame, crop with margin at native res, return
    (crops, n_fallback). Frames with no detected face -> center-square fallback."""
    pil = [Image.fromarray(f) for f in frames]
    try:
        boxes, probs = mtcnn.detect(pil)
    except Exception:
        boxes = [None] * len(pil)
    crops, n_fallback = [], 0
    for img, box in zip(pil, boxes if boxes is not None else [None] * len(pil)):
        W, H = img.size
        has = box is not None and len(box)
        if has:
            # largest by area
            b = max(box, key=lambda q: (q[2] - q[0]) * (q[3] - q[1]))
            x1, y1, x2, y2 = b
            bw, bh = x2 - x1, y2 - y1
            mx, my = bw * margin, bh * margin
            x1, y1 = max(0, x1 - mx), max(0, y1 - my)
            x2, y2 = min(W, x2 + mx), min(H, y2 + my)
            crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        else:
            n_fallback += 1
            s = min(W, H)  # center square fallback
            crop = img.crop(((W - s) // 2, (H - s) // 2,
                             (W - s) // 2 + s, (H - s) // 2 + s))
        crops.append(crop.resize((out_size, out_size), Image.BICUBIC))
    return crops, n_fallback


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
    ap.add_argument('--model', default='openai/clip-vit-large-patch14-336',
                    help='HF CLIP id; -336 = best quality (336 input)')
    ap.add_argument('--img_size', type=int, default=0,
                    help='override CLIP input size (0 = model default)')
    ap.add_argument('--max_frames', type=int, default=64)
    ap.add_argument('--margin', type=float, default=0.35)
    ap.add_argument('--batch', type=int, default=64, help='frames/CLIP forward')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--stream', action='store_true',
                    help='read clips from the remote zip via range requests '
                         '(near-zero disk); default downloads the zip once')
    ap.add_argument('--dl_dir', default='/workspace/_zip_dl')
    ap.add_argument('--limit', type=int, default=0, help='smoke: first N clips')
    ap.add_argument('--dump_crops', type=int, default=0,
                    help='save the CLIP-input crops of the first N clips to '
                         '{out_dir}/_crops/ as JPGs for a visual quality check')
    ap.add_argument('--hf_token', default=os.environ.get('HF_TOKEN') or None)
    ap.add_argument('--upload_repo', default='')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.upload_repo and not args.asset:
        do_upload(args.out_dir, args.upload_repo, args.hf_token)
        return

    import torch
    from transformers import CLIPModel, CLIPProcessor
    from facenet_pytorch import MTCNN

    dev = args.device if torch.cuda.is_available() else 'cpu'
    print(f'device={dev}  model={args.model}')
    model = CLIPModel.from_pretrained(args.model).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(args.model)
    if args.img_size:
        out_size = args.img_size
    else:
        ip = proc.image_processor
        cs = getattr(ip, 'crop_size', None) or getattr(ip, 'size', None)
        out_size = (cs.get('height') or cs.get('shortest_edge')) if isinstance(cs, dict) else int(cs)
    print(f'CLIP input size = {out_size}')
    mtcnn = MTCNN(keep_all=True, device=dev, post_process=False)

    autocast = torch.autocast(device_type='cuda', dtype=torch.float16) \
        if dev == 'cuda' else torch.autocast(device_type='cpu', enabled=False)

    done = set(os.path.splitext(f)[0] for f in os.listdir(args.out_dir)
               if f.endswith('.npy'))
    print(f'resuming: {len(done)} already extracted')

    if args.dump_crops:
        os.makedirs(os.path.join(args.out_dir, '_crops'), exist_ok=True)

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    tot_frames = tot_fallback = 0
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
            crops, nfb = crop_faces(frames, mtcnn, args.margin, out_size)
            tot_frames += len(crops); tot_fallback += nfb
            if args.dump_crops and n_ok < args.dump_crops:
                for j, c in enumerate(crops[::max(1, len(crops) // 4)][:4]):
                    c.save(os.path.join(args.out_dir, '_crops',
                                        f'{name}_{j}.jpg'))
            feats = []
            for b in range(0, len(crops), args.batch):
                batch = crops[b:b + args.batch]
                inp = proc(images=batch, return_tensors='pt').to(dev)
                with torch.no_grad(), autocast:
                    fe = model.get_image_features(**inp)  # (B, 768)
                feats.append(fe.float().cpu().numpy())
            feat = np.concatenate(feats, 0).astype(np.float32)  # (T, 768)
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

    fbrate = (tot_fallback / tot_frames * 100) if tot_frames else 0
    print(f'\nDONE: ok={n_ok} skip={n_skip} err={n_err} '
          f'in {(time.time()-t0)/60:.1f} min -> {args.out_dir}')
    print(f'no-face frames (center-crop fallback): {tot_fallback}/{tot_frames} '
          f'({fbrate:.1f}%)  [high % => MTCNN missing faces on these frames]')
    if args.upload_repo:
        do_upload(args.out_dir, args.upload_repo, args.hf_token)


if __name__ == '__main__':
    main()
