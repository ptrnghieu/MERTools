"""
mllm_dataprep.py -- build a ms-swift SFT dataset for Qwen2.5-VL on MER-Cross.

For each labelled TRAIN sample: load the OpenFace face-crop sequence, sample
n_frames evenly, save them as small JPEGs, and emit one chat example in
ms-swift JSONL format:

  {"messages": [ {system}, {user w/ <image> tags + transcript}, {assistant} ],
   "images":   ["/abs/frame0.jpg", ...]}

Modes:
  direct : assistant = '{"emotion": "<label>"}'         (fast, no teacher)
  cot    : assistant = '<reasoning> ... {"emotion":..}'  (rationalisation from a
           local Qwen served OpenAI-style; give it the TRUE label, ask it to
           justify -- label-conditioned, not circular). Requires --cot_base_url.

Role-explicit prompt: at TEST the video is the silent LISTENER and audio/text
are a different SPEAKER, so we always frame the face as the listener and the
transcript as the speaker's words (context), matching inference.
"""
import os, io, csv, json, glob, argparse
import numpy as np
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

USER_TMPL = (
    "{img_tags}\nThe images above are consecutive frames of the LISTENER's face "
    "(silent). The SPEAKER said (Chinese): \"{transcript}\".\n"
    "Predict the LISTENER's emotion. Reply ONLY with JSON: "
    '{{"emotion": "<one of neutral/angry/happy/sad/worried/surprise>"}}.'
)


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


def load_transcript_map(subtitle_csv):
    m = {}
    with open(subtitle_csv, newline='') as f:
        for row in csv.DictReader(f):
            zh = (row.get('chinese') or '').strip()
            en = (row.get('english') or '').strip()
            m[row['name']] = zh or en
    return m


def save_frames(crop_npy, name, frame_root, n_frames, quality=85):
    """Sample n_frames from the crop, save as JPEG, return list of abs paths."""
    fr = np.load(crop_npy)                       # (T,H,W,3) uint8
    out_dir = os.path.join(frame_root, name)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for j, i in enumerate(uniform_idx(len(fr), n_frames)):
        p = os.path.join(out_dir, f'{j:02d}.jpg')
        if not os.path.exists(p):
            Image.fromarray(fr[i]).convert('RGB').save(p, format='JPEG', quality=quality)
        paths.append(os.path.abspath(p))
    return paths


def cot_reason(client, model, img_paths, transcript, label):
    """Ask a local Qwen (OpenAI-compatible) to justify the KNOWN label."""
    import base64
    content = [{'type': 'text', 'text':
        f'The SPEAKER said (Chinese): "{transcript}". The frames show the silent '
        f'LISTENER. The LISTENER\'s TRUE emotion is "{label}". In 2-3 sentences, '
        f'reason step by step from the facial cues (and speaker context) to WHY the '
        f'listener feels {label}. Then end with JSON {{"emotion": "{label}"}}.'}]
    for p in img_paths:
        b = base64.b64encode(open(p, 'rb').read()).decode()
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + b}})
    r = client.chat.completions.create(model=model, temperature=0.2,
            messages=[{'role': 'system', 'content': SYSTEM},
                      {'role': 'user', 'content': content}])
    txt = r.choices[0].message.content.strip()
    if '{' not in txt:                            # ensure the JSON tail exists
        txt += f'\n{{"emotion": "{label}"}}'
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--face_root', required=True, help='openface_face dir of TRAIN crops')
    ap.add_argument('--label_csv', required=True, help='track1_train.csv (name,discrete)')
    ap.add_argument('--subtitle_csv', required=True)
    ap.add_argument('--frame_root', required=True, help='where to save sampled JPEG frames')
    ap.add_argument('--out_jsonl', required=True)
    ap.add_argument('--n_frames', type=int, default=8)
    ap.add_argument('--mode', choices=['direct', 'cot'], default='direct')
    ap.add_argument('--cot_base_url', default=None, help='OpenAI-style local Qwen endpoint (cot mode)')
    ap.add_argument('--cot_model', default='qwen2.5-vl')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    tmap = load_transcript_map(args.subtitle_csv)
    rows = list(csv.DictReader(open(args.label_csv, newline='')))
    if args.limit:
        rows = rows[:args.limit]

    client = None
    if args.mode == 'cot':
        from openai import OpenAI
        assert args.cot_base_url, 'cot mode needs --cot_base_url (serve Qwen via vLLM)'
        client = OpenAI(base_url=args.cot_base_url, api_key=os.environ.get('OPENAI_API_KEY', 'EMPTY'))

    os.makedirs(os.path.dirname(args.out_jsonl) or '.', exist_ok=True)
    n_ok, n_skip = 0, 0
    from collections import Counter
    dist = Counter()
    with open(args.out_jsonl, 'w') as fout:
        for k, row in enumerate(rows):
            name, label = row['name'], row['discrete'].strip()
            if label not in EMOS:
                n_skip += 1; continue
            crop = find_crop(args.face_root, name)
            if crop is None:
                n_skip += 1; continue
            try:
                imgs = save_frames(crop, name, args.frame_root, args.n_frames)
            except Exception as e:
                print(f'[{k}] {name} frame FAIL: {repr(e)[:80]}'); n_skip += 1; continue
            transcript = tmap.get(name, '')
            img_tags = ''.join('<image>' for _ in imgs)
            user = USER_TMPL.format(img_tags=img_tags, transcript=transcript)
            if args.mode == 'cot':
                try:
                    assistant = cot_reason(client, args.cot_model, imgs, transcript, label)
                except Exception as e:
                    print(f'[{k}] {name} cot FAIL: {repr(e)[:80]}'); n_skip += 1; continue
            else:
                assistant = json.dumps({'emotion': label})
            rec = {'messages': [{'role': 'system', 'content': SYSTEM},
                                {'role': 'user', 'content': user},
                                {'role': 'assistant', 'content': assistant}],
                   'images': imgs}
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_ok += 1; dist[label] += 1
            if k < 3 or k % 1000 == 0:
                print(f'[{k}/{len(rows)}] {name} -> {label} ({len(imgs)} frames)')

    print(f'\nDONE: {n_ok} examples ({n_skip} skipped) -> {args.out_jsonl}')
    for e in EMOS:
        print(f'  {e:10s}: {dist[e]}')


if __name__ == '__main__':
    main()
