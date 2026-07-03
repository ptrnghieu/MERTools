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


def ensemble_submission(result_npzs, save_csv):
    """Average emo_probs (logits) across multiple test1 npz files, then argmax.

    result_npzs: comma-separated list of npz paths (all must share sample order,
    which holds since test1 uses shuffle=False).
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

    emo_preds = np.argmax(avg, 1)
    emo_preds = [idx2emo_mer[idx] for idx in emo_preds]
    print(f'ensembled {len(paths)} runs, {avg.shape[0]} samples -> {save_csv}')
    _write_preds_to_csv(emo_preds, save_csv)


if __name__ == '__main__':
    import fire
    fire.Fire()
