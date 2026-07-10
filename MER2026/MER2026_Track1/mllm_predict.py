"""
MLLM-based interlocutor-emotion prediction for MER-Cross (SKETCH).

For each candidate: send the LISTENER's face frames (OpenFace crops) + the
SPEAKER's transcript to a multimodal LLM with a role-explicit dyadic prompt,
parse a 6-class score, and save emo_probs (in candidate-csv order) as an npz
compatible with submission.py (adjust_submission / ensemble_submission).

Provider-agnostic: any OpenAI-compatible chat endpoint that accepts image_url
data URIs (OpenAI, DashScope compat mode, vLLM/Qwen-VL server, ...). Audio is
NOT sent here (frames + transcript only); add it later with a native
audio+video API (Gemini / Qwen2.5-Omni).

Resumable: per-name JSON cached under cache_dir; re-run skips done names.
SMOKE TEST with --limit 5 first, inspect the printed reasons + scores, before
spending on all 20k.

Env: OPENAI_API_KEY (or provider key). --base_url / --model set the endpoint.
"""
import os, sys, glob, json, base64, argparse, io
_self = os.path.dirname(os.path.abspath(__file__))          # avoid repo statistics.py shadow
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != _self]
import numpy as np

EMOS = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']   # exact MER label vocab / idx order

SYSTEM = (
    "You are an expert FACS-based emotion reader for dyadic conversations. Two people "
    "interact: a SPEAKER (talking) and a LISTENER (silently listening). You are given frames "
    "of the LISTENER's face and the SPEAKER's transcript. Predict the LISTENER's emotion while "
    "hearing the speaker, judging PRIMARILY from the listener's facial expression; use the "
    "speaker's words only as context.\n"
    "Choose ONE of neutral, angry, happy, sad, worried, surprise using these cues:\n"
    "- neutral: relaxed face, no strong muscle action. If the expression is mild or ambiguous, "
    "choose neutral (do NOT default to worried).\n"
    "- angry: brows LOWERED and drawn together, hard/tense stare or glare, lips pressed/tightened, "
    "jaw tension. Lowered tense brows = angry, NOT worried.\n"
    "- happy: smile, raised cheeks, crow's-feet.\n"
    "- sad: INNER brow corners raised, lip corners pulled DOWN, downcast/drooping look.\n"
    "- worried: INNER brows raised with anxious concern, NO aggression or glare. Use worried ONLY "
    "when clear anxiety is visible, NOT for any furrowed brow.\n"
    "- surprise: BOTH brows raised high, eyes widened, jaw dropped (brief). Do not confuse mild "
    "attentiveness with surprise.\n"
    "KEY: distinguish ANGRY (lowered, tense, aggressive brows) from WORRIED (raised inner brows, "
    "anxious, non-aggressive). Anger is common; do not avoid it.\n"
    'Reply ONLY with JSON: {"reason": "<short>", "scores": {"neutral":0-100, "angry":0-100, '
    '"happy":0-100, "sad":0-100, "worried":0-100, "surprise":0-100}} with scores summing to 100.'
)


def uniform_idx(n, k):
    if n <= 0: return [0] * k
    if n <= k: return list(range(n)) + [n - 1] * (k - n)
    return np.linspace(0, n - 1, k).astype(int).tolist()


def find_crop(face_root, name):
    for cand in [os.path.join(face_root, name, name + '.npy'),
                 os.path.join(face_root, name + '.npy')]:
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(face_root, name, '*.npy'))   # any npy in the name dir
    return hits[0] if hits else None


def frames_to_data_uris(crop_npy, n_frames):
    from PIL import Image
    fr = np.load(crop_npy)                                   # (T,H,W,3) uint8
    uris = []
    for i in uniform_idx(len(fr), n_frames):
        buf = io.BytesIO(); Image.fromarray(fr[i]).save(buf, format='JPEG', quality=85)
        uris.append('data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode())
    return uris


def parse_scores(text):
    import re
    v = None
    try:
        s = text[text.find('{'): text.rfind('}') + 1]
        d = json.loads(s)['scores']
        v = np.array([float(d.get(e, 0)) for e in EMOS], dtype=np.float64)
    except Exception:
        # robust fallback: regex "<label>": <number> even if JSON is malformed
        v = np.zeros(6)
        for i, e in enumerate(EMOS):
            m = re.search(rf'"{e}"\s*:\s*([0-9.]+)', text)
            if m:
                v[i] = float(m.group(1))
    if v is None or v.sum() <= 0:
        v = np.ones(6)
    return v / v.sum()                                       # probs over 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--face_root', required=True, help='openface_face dir ({name}/{name}.npy crops)')
    ap.add_argument('--subtitle_csv', required=True, help='name -> chinese/english transcript')
    ap.add_argument('--candidate_csv', required=True, help='track1_track2_candidate.csv (order)')
    ap.add_argument('--out_npz', required=True)
    ap.add_argument('--cache_dir', default='./mllm_cache')
    ap.add_argument('--base_url', default=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    ap.add_argument('--model', default='gpt-4o-mini')
    ap.add_argument('--n_frames', type=int, default=6)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=os.environ['OPENAI_API_KEY'])
    os.makedirs(args.cache_dir, exist_ok=True)

    names = pd.read_csv(args.candidate_csv)['name'].astype(str).tolist()
    sub = pd.read_csv(args.subtitle_csv).set_index('name')
    def transcript(n):
        if n not in sub.index: return ''
        row = sub.loc[n]
        return str(row.get('chinese', '') or '') or str(row.get('english', '') or '')

    if args.limit: names = names[:args.limit]
    probs_all, n_fail = [], 0
    for k, name in enumerate(names):
        cache = os.path.join(args.cache_dir, name + '.json')
        if os.path.exists(cache):
            probs_all.append(np.array(json.load(open(cache))['probs'])); continue
        crop = find_crop(args.face_root, name)
        try:
            if crop is None:
                raise FileNotFoundError(f'no crop npy for {name}')
            uris = frames_to_data_uris(crop, args.n_frames)
            content = [{'type': 'text',
                        'text': f'Speaker said (Chinese): "{transcript(name)}". '
                                f'Here are {len(uris)} frames of the LISTENER. Predict the LISTENER\'s emotion.'}]
            content += [{'type': 'image_url', 'image_url': {'url': u}} for u in uris]
            r = client.chat.completions.create(model=args.model, temperature=0,
                    messages=[{'role': 'system', 'content': SYSTEM},
                              {'role': 'user', 'content': content}])
            txt = r.choices[0].message.content
            p = parse_scores(txt)
        except Exception as e:
            print(f'[{k}] {name} FAIL: {repr(e)[:120]}')
            probs_all.append(np.ones(6) / 6); n_fail += 1
            continue                                          # do NOT cache failures
        json.dump({'probs': p.tolist(), 'raw': txt}, open(cache, 'w'))
        probs_all.append(p)
        if k < 5 or k % 500 == 0:
            print(f'[{k}/{len(names)}] {name} -> {EMOS[int(p.argmax())]} {p.round(2)}')

    probs = np.stack(probs_all)
    logits = np.log(probs + 1e-9)                            # pseudo-logits for submission.py
    np.savez_compressed(args.out_npz, emo_probs=logits)
    print(f'DONE: {len(names)} preds ({n_fail} failed->uniform) -> {args.out_npz}')
    from collections import Counter
    c = Counter(probs.argmax(1).tolist())
    for i in range(6): print(f'  {EMOS[i]:10s}: {c[i]}')


if __name__ == '__main__':
    main()
