"""
Cheap validation: run the MLLM predictor on a labeled TRAIN subset and compute
WAF directly against ground truth (no Codabench needed). Reuses the exact
prompt/parse/crop logic from mllm_predict.

CAVEAT: train is the INDIVIDUAL domain (video, audio, text are the SAME person),
so this measures the MLLM's raw face+context emotion reading -- an upper-ish
PROXY for the cross-role test, not the exact test WAF. High here => worth the
full 20k candidate run; low here => stop, save the money.

Usage:
  export OPENAI_API_KEY=sk-or-...
  python3 eval_mllm_train.py \
    --face_root /workspace/mer2026/raw_face/openface_face \
    --subtitle_csv /workspace/mer2026/raw_dl/subtitle_chieng.csv \
    --train_csv /workspace/mer2026/raw_dl/track1_train.csv \
    --n 500 --model google/gemini-2.5-flash
"""
import os, sys, json, argparse, random
_self = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != _self]
import numpy as np
from mllm_predict import SYSTEM, EMOS, find_crop, frames_to_data_uris, parse_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--face_root', required=True)
    ap.add_argument('--subtitle_csv', required=True)
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--base_url', default='https://openrouter.ai/api/v1')
    ap.add_argument('--model', default='google/gemini-2.5-flash')
    ap.add_argument('--n_frames', type=int, default=6)
    ap.add_argument('--cache_dir', default='./mllm_train_cache')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import pandas as pd
    from openai import OpenAI
    from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

    client = OpenAI(base_url=args.base_url, api_key=os.environ['OPENAI_API_KEY'])
    os.makedirs(args.cache_dir, exist_ok=True)

    df = pd.read_csv(args.train_csv)                       # name, discrete
    bad = df[~df['discrete'].isin(EMOS)]['discrete'].unique()
    if len(bad):
        print(f'WARN: labels not in EMOS (dropped): {list(bad)[:10]}')
    df = df[df['discrete'].isin(EMOS)].reset_index(drop=True)
    random.seed(args.seed)
    idx = list(range(len(df))); random.shuffle(idx); rows = df.iloc[idx[:args.n]]

    sub = pd.read_csv(args.subtitle_csv).set_index('name')
    def transcript(n):
        if n not in sub.index: return ''
        r = sub.loc[n]; return str(r.get('chinese', '') or '') or str(r.get('english', '') or '')

    emo2idx = {e: i for i, e in enumerate(EMOS)}
    preds, gts, nfail = [], [], 0
    for k, (_, row) in enumerate(rows.iterrows()):
        name = str(row['name']); gt = emo2idx[row['discrete']]
        cache = os.path.join(args.cache_dir, name + '.json')
        if os.path.exists(cache):
            p = np.array(json.load(open(cache))['probs'])
        else:
            crop = find_crop(args.face_root, name)
            try:
                if crop is None:
                    raise FileNotFoundError(name)
                uris = frames_to_data_uris(crop, args.n_frames)
                content = [{'type': 'text',
                            'text': f'Speaker said (Chinese): "{transcript(name)}". '
                                    f'Here are {len(uris)} frames of the LISTENER. Predict the LISTENER\'s emotion.'}]
                content += [{'type': 'image_url', 'image_url': {'url': u}} for u in uris]
                r = client.chat.completions.create(model=args.model, temperature=0,
                        messages=[{'role': 'system', 'content': SYSTEM},
                                  {'role': 'user', 'content': content}])
                txt = r.choices[0].message.content; p = parse_scores(txt)
                json.dump({'probs': p.tolist(), 'raw': txt}, open(cache, 'w'))
            except Exception as e:
                print(f'[{k}] {name} FAIL {repr(e)[:100]}'); nfail += 1; continue
        preds.append(int(p.argmax())); gts.append(gt)
        if k < 3 or k % 100 == 0:
            print(f'[{k}/{len(rows)}] {name} pred={EMOS[preds[-1]]} gt={EMOS[gt]}')

    preds, gts = np.array(preds), np.array(gts)
    L = list(range(6))
    print(f'\nevaluated {len(preds)} (failed {nfail})')
    print('gt distribution :', {EMOS[i]: int((gts == i).sum()) for i in L})
    print('WAF (weighted-F1):', round(f1_score(gts, preds, average='weighted', labels=L, zero_division=0), 4))
    print('macro-F1        :', round(f1_score(gts, preds, average='macro', labels=L, zero_division=0), 4))
    print('accuracy        :', round(accuracy_score(gts, preds), 4))
    print()
    print(classification_report(gts, preds, labels=L, target_names=EMOS, digits=3, zero_division=0))
    print('confusion (rows=gt, cols=pred):')
    print('        ' + '  '.join(f'{e[:4]:>4}' for e in EMOS))
    for i, r in enumerate(confusion_matrix(gts, preds, labels=L)):
        print(f'  {EMOS[i][:4]:>4}  ' + '  '.join(f'{v:>4}' for v in r))


if __name__ == '__main__':
    main()
