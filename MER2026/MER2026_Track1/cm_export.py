"""
Export in-domain confusion matrices + per-class P/R/F1 for every trained
architecture that has a cv_*.npz (eval_emo_probs / eval_emo_labels). Picks the
best-WAF cv per (model, video-feature) and prints ONE compact JSON blob (paste
it back to render heatmaps). Standalone: numpy + sklearn only (no toolkit).

  python3 cm_export.py > cm.json      # then paste cm.json
"""
import glob, re, json
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score

EMOS = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']


def feat_key(f):
    if 'hsemotion' in f: return 'FER'
    if 'clip_au20' in f: return 'CLIP+AU'
    return 'CLIP'


best = {}
for f in glob.glob('saved-trimodal/result/cv_*.npz'):
    try:
        d = np.load(f, allow_pickle=True)
    except Exception:
        continue
    if 'eval_emo_probs' not in d or len(d['eval_emo_probs']) == 0:
        continue
    y = np.array(d['eval_emo_labels'].tolist()).astype(int)
    if len(y) > 9600:                      # skip pseudo-label-mixed run (inflated)
        continue
    p = np.array(d['eval_emo_probs'].tolist()).argmax(1)
    waf = f1_score(y, p, average='weighted')
    m = re.search(r'model:([^+]+)', f).group(1)
    k = (m, feat_key(f))
    if k not in best or waf > best[k][0]:
        best[k] = (waf, y, p)

out = []
for (m, vf), (waf, y, p) in best.items():
    cm = confusion_matrix(y, p, labels=range(6))
    P, R, F, _ = precision_recall_fscore_support(y, p, labels=range(6), zero_division=0)
    name = m.replace('memocmt_', '') if vf == 'CLIP' else f"{m.replace('memocmt_','')} ({vf})"
    out.append({
        'name': name, 'waf': round(float(waf), 4),
        'cm': cm.tolist(),
        'per': {EMOS[i]: {'p': round(float(P[i]), 3), 'r': round(float(R[i]), 3),
                          'f1': round(float(F[i]), 3), 'n': int(cm[i].sum())}
                for i in range(6)},
    })
out.sort(key=lambda x: -x['waf'])
print(json.dumps(out))
