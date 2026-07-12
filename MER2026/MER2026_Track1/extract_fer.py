"""
extract_fer.py -- per-frame FER-specialized features for the video branch.

Replaces generic CLIP video features with an AffectNet-pretrained facial-
expression backbone (HSEmotion / Savchenko), whose feature space is organized
BY expression -> separates emotions far better than CLIP and transfers across
roles (a frown is a frown, speaking or listening). This directly targets the
bottleneck confirmed twice (MLLM's weak generic vision; ensemble flat): video
representation quality.

Output matches the toolkit feature format exactly -- one npy per sample,
shape (n_frames, fer_dim), at {out_dir}/{name}.npy -- so main-release.py reads
it as --video_feature <name> with feat_type frm_unalign (loader compresses /12).

Run once per crop set (train, candidate) into the SAME out_dir (names don't
collide: sample_* vs samplenew_*). Then copy out_dir to the v3 instance under
PATH_TO_FEATURES and retrain.

Env: pip install hsemotion timm   (torch already present)

  python extract_fer.py \
    --face_root /workspace/crops_train/openface_face \
    --out_dir   /workspace/fer_feats/hsemotion-enet_b2_8-FRA \
    --model_name enet_b2_8 --max_frames 64 --limit 20     # SMOKE first
"""
import os, sys, glob, argparse
# repo ships a statistics.py that shadows stdlib `statistics`; when torch's
# inductor does `import statistics` it grabs the repo file (which imports
# torchaudio) -> spurious ModuleNotFoundError. Strip the repo dir from sys.path
# BEFORE importing torch/torchvision/hsemotion.
_self = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != _self]
import numpy as np


def uniform_idx(n, k):
    if n <= 0: return [0] * k
    if n <= k: return list(range(n))
    return np.linspace(0, n - 1, k).astype(int).tolist()


def find_crop(face_root, name):
    for c in [os.path.join(face_root, name, name + '.npy'),
              os.path.join(face_root, name + '.npy')]:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(face_root, name, '*.npy'))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--face_root', required=True, help='openface_face dir ({name}/{name}.npy)')
    ap.add_argument('--out_dir', required=True, help='feature output dir (= {feat_root}/{name}-FRA)')
    ap.add_argument('--model_name', default='enet_b2_8',
                    help='HSEmotion model (enet_b2_8=1408d AffectNet-8; enet_b0_8=1280d lighter)')
    ap.add_argument('--max_frames', type=int, default=64,
                    help='uniform frames/clip to keep (loader then compresses /12 ~ matches CLIP T)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    # torch>=2.6 defaults weights_only=True; HSEmotion ships full pickled models
    # and calls torch.load without the flag -> force weights_only=False (trusted).
    import torch, functools
    _orig_load = torch.load
    torch.load = functools.partial(_orig_load, weights_only=False)

    from hsemotion.facial_emotions import HSEmotionRecognizer
    from PIL import Image
    fer = HSEmotionRecognizer(model_name=args.model_name, device=args.device)
    torch.load = _orig_load                         # restore
    fer.model.eval()
    tf = fer.test_transforms                         # Resize(260)+ToTensor+Normalize

    os.makedirs(args.out_dir, exist_ok=True)
    names = sorted(d for d in os.listdir(args.face_root)
                   if os.path.isdir(os.path.join(args.face_root, d)))
    if args.limit:
        names = names[:args.limit]

    done, skip, dim = 0, 0, None
    for k, name in enumerate(names):
        out = os.path.join(args.out_dir, name + '.npy')
        if os.path.exists(out):
            skip += 1; continue
        crop = find_crop(args.face_root, name)
        if crop is None:
            print(f'[{k}] {name} no crop'); continue
        fr = np.load(crop)                                  # (T,H,W,3) uint8 RGB
        idxs = uniform_idx(len(fr), min(args.max_frames, len(fr)))
        with torch.no_grad():                               # batch ALL frames -> 1 forward
            batch = torch.stack([tf(Image.fromarray(fr[i]).convert('RGB')) for i in idxs]).to(args.device)
            arr = fer.model(batch).float().cpu().numpy().astype(np.float32)   # (n, fer_dim)
        np.save(out, arr)
        done += 1; dim = arr.shape[1]
        if k < 3 or k % 1000 == 0:
            print(f'[{k}/{len(names)}] {name} -> {arr.shape}')
    print(f'\nDONE: {done} extracted, {skip} skipped -> {args.out_dir}  (fer_dim={dim})')


if __name__ == '__main__':
    main()
