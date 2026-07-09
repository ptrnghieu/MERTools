import itertools
import torchaudio

from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
import numpy as np

from toolkit.globals import *
from toolkit.utils.functions import *
from sklearn.metrics import f1_score, accuracy_score


def _write_preds_to_csv(emo_preds, save_csv):
    # names from the candidate csv (w/o gt), preserving order
    label_csv = os.path.join(config.DATA_DIR['MER2026'], 'track1_track2_candidate.csv')
    names = func_read_key_from_csv(label_csv, 'name')
    name2key = {}
    for (name, pred) in zip(names, emo_preds):
        name2key[name] = pred
    func_write_key_to_csv(save_csv, names, name2key, ['discrete'])


def generate_submission(result_npz, save_csv):

    # 1. read preds
    emo_probs = np.load(result_npz, allow_pickle=True)['emo_probs'].tolist()
    emo_preds = np.argmax(emo_probs, 1)
    emo_preds = [idx2emo_mer[idx] for idx in emo_preds]

    # 2+3. names + save_csv
    _write_preds_to_csv(emo_preds, save_csv)


def _train_prior():
    """Class prior over the 6 emotions, in idx order, from the train corpus."""
    corpus = np.load(config.PATH_TO_LABEL['MER2026'], allow_pickle=True)['train_corpus'].tolist()
    counts = np.zeros(len(idx2emo_mer), dtype=np.float64)
    for v in corpus.values():
        counts[emo2idx_mer[v['emo']]] += 1
    return counts / counts.sum()


def adjust_submission(result_npz, save_csv, tau=1.0):
    """Post-hoc label-shift correction: logit_adj = logit - tau * log(train_prior).

    tau=0 -> original argmax; tau=1 -> full shift toward uniform target;
    tau>1 -> push further toward rare classes. No retraining. Prints the
    resulting predicted class distribution so tau can be chosen by inspection.
    """
    logits = np.array(np.load(result_npz, allow_pickle=True)['emo_probs'].tolist(), dtype=np.float64)
    log_prior = np.log(_train_prior() + 1e-12)
    adj = logits - tau * log_prior[None, :]

    preds_idx = np.argmax(adj, 1)
    n = len(preds_idx)
    from collections import Counter
    c = Counter(preds_idx.tolist())
    print(f'tau={tau} -> predicted distribution ({n} samples):')
    for idx in sorted(c):
        print(f'   {idx2emo_mer[idx]:10s}: {c[idx]:6d} ({100*c[idx]/n:5.1f}%)')

    emo_preds = [idx2emo_mer[idx] for idx in preds_idx]
    _write_preds_to_csv(emo_preds, save_csv)


def val_report(cv_npz, tau=0.0):
    """Confusion matrix + per-class precision/recall/F1/acc on the validation
    folds, from a cv npz that stored eval_emo_probs / eval_emo_labels.
    tau optionally applies the same label-shift used at test (default off; on
    val the prior is the train prior so tau!=0 is only for what-if inspection).
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    d = np.load(cv_npz, allow_pickle=True)
    if 'eval_emo_probs' not in d or len(d['eval_emo_probs']) == 0:
        print('This cv npz has no eval predictions (older run). Re-run with the patched code.')
        return
    probs  = np.array(d['eval_emo_probs'].tolist(), dtype=np.float64)
    labels = np.array(d['eval_emo_labels'].tolist()).astype(int)
    if tau != 0.0:
        probs = probs - tau * np.log(_train_prior() + 1e-12)[None, :]
    preds = probs.argmax(1)
    names = [idx2emo_mer[i] for i in range(len(idx2emo_mer))]
    print(f'val samples: {len(labels)} | overall acc: {accuracy_score(labels, preds):.4f} | tau={tau}\n')
    print('Confusion matrix (rows=true, cols=pred):')
    print('        ' + '  '.join(f'{n[:4]:>4}' for n in names))
    for i, row in enumerate(confusion_matrix(labels, preds, labels=list(range(len(names))))):
        print(f'  {names[i][:4]:>4}  ' + '  '.join(f'{v:>4}' for v in row))
    print()
    print(classification_report(labels, preds, target_names=names, digits=4))


def ensemble_submission(result_npzs, save_csv, tau=0.0):
    """Average emo_probs (logits) across multiple test1 npz files, apply optional
    label-shift correction (tau), then argmax.

    result_npzs: comma-separated list of npz paths (all must share sample order,
    which holds since test1 uses shuffle=False).
    tau: post-hoc label-shift strength applied to the averaged logits.
    """
    paths = [p.strip() for p in result_npzs.split(',') if p.strip()]
    assert len(paths) >= 1, 'need at least one npz'

    stacked = None
    for p in paths:
        probs = np.array(np.load(p, allow_pickle=True)['emo_probs'].tolist(), dtype=np.float64)
        assert stacked is None or probs.shape == stacked.shape, \
            f'shape mismatch: {p} -> {probs.shape} vs {stacked.shape}'
        stacked = probs if stacked is None else stacked + probs
    avg = stacked / len(paths)

    if tau != 0.0:
        avg = avg - tau * np.log(_train_prior() + 1e-12)[None, :]

    preds_idx = np.argmax(avg, 1)
    n = len(preds_idx)
    from collections import Counter
    c = Counter(preds_idx.tolist())
    print(f'ensembled {len(paths)} runs (tau={tau}), {n} samples:')
    for idx in sorted(c):
        print(f'   {idx2emo_mer[idx]:10s}: {c[idx]:6d} ({100*c[idx]/n:5.1f}%)')

    emo_preds = [idx2emo_mer[idx] for idx in preds_idx]
    _write_preds_to_csv(emo_preds, save_csv)


def make_pseudo_corpus(result_npz, save_npz, tau=1.5, top_frac=0.5, min_conf=0.0):
    """Transductive self-training: turn a teacher model's predictions on the
    dyadic candidate pool into a pseudo-labeled training corpus, so a retrained
    model can actually SEE (listener-video + speaker-audio/text) examples and
    learn the speaker->listener relationship that Individual training lacks.

    Principled guards against confirmation bias (amplifying the train prior):
      1. Pseudo-labels come from tau-ADJUSTED logits (logit - tau*log train_prior),
         i.e. the same label-shift-corrected predictions we submit -- so the
         pseudo distribution approximates the TEST prior, not the train prior.
      2. Per-class top-fraction selection: within each predicted class we keep
         only the most-confident `top_frac`; this preserves the (corrected)
         class balance instead of letting majority classes dominate.
      3. `min_conf` floor drops low-confidence garbage.

    result_npz : teacher test1 npz (emo_probs = logits over the candidate pool,
                 in candidate-csv order). Use the strongest single model (v3) or
                 an ensembled-logits npz.
    save_npz   : output label npz {train_corpus (real+pseudo), test1_corpus}.
    """
    from collections import Counter, defaultdict

    # real corpus (train + candidate order) from the current label file
    label_path   = config.PATH_TO_LABEL['MER2026']
    d0           = np.load(label_path, allow_pickle=True)
    train_corpus = dict(d0['train_corpus'].tolist())
    test1_corpus = dict(d0['test1_corpus'].tolist())

    cand_csv = os.path.join(config.DATA_DIR['MER2026'], 'track1_track2_candidate.csv')
    cand_names = func_read_key_from_csv(cand_csv, 'name')

    logits = np.array(np.load(result_npz, allow_pickle=True)['emo_probs'].tolist(), dtype=np.float64)
    assert len(logits) == len(cand_names), \
        f'logits ({len(logits)}) != candidate names ({len(cand_names)})'

    # tau-adjusted softmax -> pseudo label + confidence (test-prior calibrated)
    adj  = logits - tau * np.log(_train_prior() + 1e-12)[None, :]
    adj -= adj.max(1, keepdims=True)
    probs = np.exp(adj); probs /= probs.sum(1, keepdims=True)
    pseudo = probs.argmax(1)
    conf   = probs.max(1)

    # per-class top-fraction selection (preserves corrected class balance)
    by_class = defaultdict(list)
    for i in range(len(cand_names)):
        if conf[i] >= min_conf:
            by_class[int(pseudo[i])].append((conf[i], i))

    selected = []
    for c, items in by_class.items():
        items.sort(reverse=True)                    # most confident first
        keep = max(1, int(round(len(items) * top_frac)))
        selected.extend(idx for _, idx in items[:keep])

    added = Counter()
    n_skip_collision = 0
    for i in selected:
        name = cand_names[i]
        if name in train_corpus:                    # never overwrite a real label
            n_skip_collision += 1
            continue
        train_corpus[name] = {'emo': idx2emo_mer[int(pseudo[i])]}
        added[int(pseudo[i])] += 1

    # report
    n_real   = len(d0['train_corpus'].tolist())
    n_pseudo = sum(added.values())
    print(f'teacher npz: {result_npz}')
    print(f'tau={tau}  top_frac={top_frac}  min_conf={min_conf}')
    print(f'candidates: {len(cand_names)} | selected: {len(selected)} | '
          f'added (new names): {n_pseudo} | collisions skipped: {n_skip_collision}')
    print(f'corpus: {n_real} real -> {len(train_corpus)} total (+{n_pseudo} pseudo)\n')
    print(f'{"class":10s} {"raw-argmax":>11s} {"tau-argmax":>11s} {"pseudo-added":>12s} {"real-train":>11s}')
    raw_dist  = Counter(logits.argmax(1).tolist())   # tau=0 genuine belief
    cand_dist = Counter(pseudo.tolist())             # tau-adjusted argmax
    real_dist = Counter(emo2idx_mer[v["emo"]] for v in d0['train_corpus'].tolist().values())
    for idx in range(len(idx2emo_mer)):
        print(f'{idx2emo_mer[idx]:10s} {raw_dist[idx]:11d} {cand_dist[idx]:11d} '
              f'{added[idx]:12d} {real_dist[idx]:11d}')
    print('\nCHECK: if pseudo-added for a rare class >> its raw-argmax count, tau is\n'
          'force-labeling low-belief samples -> lower tau (try 1.0) before retraining.')

    np.savez_compressed(save_npz, train_corpus=train_corpus, test1_corpus=test1_corpus)
    print(f'\nsaved pseudo corpus -> {save_npz}')
    print('next: point config.PATH_TO_LABEL[MER2026] at this npz, retrain, re-sweep tau.')


if __name__ == '__main__':
    import fire
    fire.Fire()
