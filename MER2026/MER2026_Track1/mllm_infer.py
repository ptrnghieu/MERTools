"""
mllm_infer.py -- run the fine-tuned Qwen2.5-VL on the candidate pool and emit
soft 6-way probabilities via VERBALIZER SCORING (not free generation).

For each candidate: build the same role-explicit prompt (listener frames +
speaker transcript), then for each of the 6 emotion words score the model's
log-likelihood of that word completing '{"emotion": "<word>"}'. Softmax over the
6 label log-probs -> a calibrated distribution. Saved as emo_probs = log(probs)
(pseudo-logits) in candidate-csv order, so submission.py / mlls_submission /
adjust_submission consume it unchanged.

Verbalizer scoring (vs argmax generation) gives the soft probabilities MLLS/EM
label-shift needs, and avoids brittle JSON parsing.

  python mllm_infer.py \
    --base Qwen/Qwen2.5-VL-7B-Instruct \
    --adapter /workspace/mllm_sft_direct/vX-.../checkpoint-XXX \
    --face_root /workspace/crops_cand/openface_face \
    --candidate_csv /workspace/mer2026_meta/track1_track2_candidate.csv \
    --subtitle_csv  /workspace/mer2026_meta/subtitle_chieng.csv \
    --out_npz /workspace/mllm_out/test1_mllm_direct.npz \
    --n_frames 8 --limit 20      # SMOKE first
"""
import os, csv, glob, argparse
import numpy as np
import torch
from PIL import Image

EMOS = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']

SYSTEM = (
    "You are an expert FACS-based emotion reader for dyadic conversations. Two people "
    "interact: a SPEAKER (talking) and a LISTENER (silently listening). You are shown "
    "frames of the LISTENER's face and the SPEAKER's transcript. Judge the LISTENER's "
    "emotion PRIMARILY from their facial expression; use the speaker's words only as "
    "context (emotional contagion). Choose exactly one of: "
    "neutral, angry, happy, sad, worried, surprise."
)
HEAD = ("The images above are consecutive frames of the LISTENER's face (silent). ")
CTX_WITH = 'The SPEAKER said (Chinese): "{t}". '
CTX_NONE = 'No speaker transcript is available. '
TAIL = ("Predict the LISTENER's emotion. Reply ONLY with JSON: "
        '{"emotion": "<one of neutral/angry/happy/sad/worried/surprise>"}.')
ASST_PREFIX = '{"emotion": "'          # completion up to the label word


def uniform_idx(n, k):
    if n <= 0: return [0] * k
    if n <= k: return list(range(n)) + [n - 1] * (k - n)
    return np.linspace(0, n - 1, k).astype(int).tolist()


def find_crop(face_root, name):
    for c in [os.path.join(face_root, name, name + '.npy'),
              os.path.join(face_root, name + '.npy')]:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(face_root, name, '*.npy'))
    return hits[0] if hits else None


def load_frames(crop_npy, n):
    fr = np.load(crop_npy)                          # (T,H,W,3) uint8
    return [Image.fromarray(fr[i]).convert('RGB') for i in uniform_idx(len(fr), n)]


def transcript_map(subtitle_csv):
    m = {}
    with open(subtitle_csv, newline='') as f:
        for r in csv.DictReader(f):
            m[r['name']] = (r.get('chinese') or '').strip() or (r.get('english') or '').strip()
    return m


@torch.no_grad()
def score_labels(model, processor, frames, transcript, first_ids):
    """First-token verbalizer scoring: ONE forward over the prompt ending in
    '{"emotion": "', read the next-token log-prob of each label's first token
    (all 6 are distinct), softmax. This is exactly what the model would
    generate, needs one forward (not six), and avoids the multi-token
    mean-logprob bias that suppressed single-token labels (e.g. happy)."""
    ctx = CTX_WITH.format(t=transcript) if transcript.strip() else CTX_NONE
    user_text = HEAD + ctx + TAIL
    messages = [
        {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
        {'role': 'user', 'content': [{'type': 'image', 'image': im} for im in frames]
                                    + [{'type': 'text', 'text': user_text}]},
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = prompt + ASST_PREFIX
    inputs = processor(text=[prefix], images=frames, return_tensors='pt').to(model.device)
    out = model(**inputs)
    logp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)   # next-token dist
    s = np.array([float(logp[i]) for i in first_ids], dtype=np.float64)
    z = s - s.max()
    p = np.exp(z); p /= p.sum()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--adapter', required=True, help='LoRA checkpoint dir from swift sft')
    ap.add_argument('--face_root', required=True, help='candidate openface_face dir')
    ap.add_argument('--candidate_csv', required=True)
    ap.add_argument('--subtitle_csv', required=True)
    ap.add_argument('--out_npz', required=True)
    ap.add_argument('--n_frames', type=int, default=8)
    ap.add_argument('--max_pixels', type=int, default=112 * 112)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(args.base, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map='cuda')
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    first_ids = [processor.tokenizer(e, add_special_tokens=False).input_ids[0] for e in EMOS]
    assert len(set(first_ids)) == len(EMOS), f'label first-tokens collide: {first_ids}'
    print('label first-token ids:', dict(zip(EMOS, first_ids)))

    names = [r['name'] for r in csv.DictReader(open(args.candidate_csv, newline=''))]
    if args.limit:
        names = names[:args.limit]
    tmap = transcript_map(args.subtitle_csv)

    probs_all, n_fail = [], 0
    for k, name in enumerate(names):
        crop = find_crop(args.face_root, name)
        try:
            if crop is None:
                raise FileNotFoundError(f'no crop for {name}')
            frames = load_frames(crop, args.n_frames)
            p = score_labels(model, processor, frames, tmap.get(name, ''), first_ids)
        except Exception as e:
            print(f'[{k}] {name} FAIL: {repr(e)[:100]}')
            p = np.ones(6) / 6; n_fail += 1
        probs_all.append(p)
        if k < 5 or k % 500 == 0:
            print(f'[{k}/{len(names)}] {name} -> {EMOS[int(p.argmax())]} {p.round(3)}')

    probs = np.stack(probs_all)
    os.makedirs(os.path.dirname(args.out_npz) or '.', exist_ok=True)
    np.savez_compressed(args.out_npz, emo_probs=np.log(probs + 1e-9))
    from collections import Counter
    c = Counter(probs.argmax(1).tolist())
    print(f'\nDONE: {len(probs)} preds ({n_fail} failed) -> {args.out_npz}')
    for i in range(6):
        print(f'  {EMOS[i]:10s}: {c[i]}')
    print('\nnext: submission.py mlls_submission / adjust_submission on this npz')


if __name__ == '__main__':
    main()
