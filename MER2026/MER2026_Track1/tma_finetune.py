"""
tma_finetune.py -- BBA-style Target-related Model Adaptation for MER-Cross.

Idea (from "Bridge Then Begin Anew", AAAI-25): a source model overfit to the
train (Individual) distribution keeps talking-face bias that hurts on the test
(Interlocutor) distribution where the video is a SILENT listener. Instead of
fine-tuning the whole thing, keep the source model as a BRIDGE (frozen speaker
knowledge + pseudo-labeller) and RE-INITIALISE only the listener temporal
pooling + fusion head, then re-learn them PURELY on the target (candidate)
distribution via the bridge's tau-calibrated pseudo-labels.

Two modes (main-release.py / memocmt_v1.py / memocmt_v3.py are NOT touched):

  --mode bridge : train ONE MemoCMTV3 on the REAL train corpus, save its
                  state_dict (for the frozen speaker/video weights used in TMA)
                  and dump test1 logits over the candidate pool (pseudo source).

  --mode tma    : load the bridge, FREEZE {audio/text/video encoders, speaker
                  cross-attn}, RE-INIT {listener_pool, cross_attn_fusion,
                  norm_fusion, fc_out_1/2}, and adapt ONLY those on the
                  candidate pool (pseudo-labelled by --bridge_logits,
                  tau-calibrated + per-class top-fraction). Writes a
                  test1_*.npz in the SAME format main-release.py produces, so
                  submission.py / ensemble_submission consume it unchanged.

Tier-1 (default): hard pseudo-CE, no clustering label-refinement, no KL. This
probes the single hypothesis "does refitting the head to the listener feature
geometry help?". If it beats 65.78 by >=1 WAF, add DMG clustering (Tier-2).
"""
import os
import time
import random
import argparse
import numpy as np
from collections import Counter, defaultdict
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from toolkit.globals import *
from toolkit.utils.loss import CELoss
from toolkit.utils.functions import *
from toolkit.models import get_models
from toolkit.models.memocmt_v3 import LearnableQueryPooling
from toolkit.dataloader import get_dataloaders

# module attributes on MemoCMTV3 (see memocmt_v3.py)
SPEAKER_MODULES = ['audio_encoder', 'text_encoder',
                   'cross_attn_a2t', 'proj_a2t', 'norm_a2t',
                   'cross_attn_t2a', 'proj_t2a', 'norm_t2a', 'speaker_drop']
REINIT_MODULES  = ['listener_pool', 'cross_attn_fusion', 'norm_fusion',
                   'fc_out_1', 'fc_out_2']


# --------------------------------------------------------------------------- #
#  shared helpers
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def prep_feat_scale(args):
    if args.feat_type == 'utt':
        args.feat_scale = 1
    elif args.feat_type == 'frm_align':
        args.feat_scale = 6
    elif args.feat_type == 'frm_unalign':
        args.feat_scale = 12
    for f in [args.audio_feature, args.text_feature, args.video_feature]:
        assert f is not None and f.endswith('FRA'), 'TMA expects frame-level (FRA) features'


def class_weights_from_labels(labels_idx, n_cls):
    counts = np.bincount(np.asarray(labels_idx, dtype=int), minlength=n_cls).astype(float)
    counts = np.where(counts == 0, 1, counts)
    w = len(labels_idx) / (n_cls * counts)
    return torch.tensor(w, dtype=torch.float).cuda()


def run_epoch(args, model, cls_loss, loader, optimizer=None, train=False, frozen=None):
    inner = model.module if hasattr(model, 'module') else model
    config.train = train
    if train:
        model.train()
        # keep frozen extractors deterministic (no dropout / fixed behaviour)
        for n in (frozen or []):
            getattr(inner.model, n).eval()
    else:
        model.eval()

    all_probs, losses = [], []
    for data in loader:
        batch, emos, vals, bnames = data
        for k in batch: batch[k] = batch[k].cuda()
        emos = emos.cuda()
        batch['emos'] = emos
        if train:
            optimizer.zero_grad()
        features, emos_out, vals_out, interloss = model(batch)
        if train:
            loss = interloss.mean() + cls_loss(emos_out, emos)
            loss.backward()
            if inner.model.grad_clip != -1:
                torch.nn.utils.clip_grad_value_(
                    [p for p in model.parameters() if p.requires_grad], inner.model.grad_clip)
            optimizer.step()
            losses.append(float(loss.data.cpu().numpy()))
        else:
            all_probs.append(emos_out.data.cpu().numpy())
    if train:
        return np.mean(losses) if losses else 0.0
    return np.concatenate(all_probs) if all_probs else np.zeros((0, args.output_dim1))


def build_model_and_loaders(args):
    dl = get_dataloaders(args)                       # sets output_dim1=6, output_dim2=0
    train_loaders, eval_loaders, test_loaders = dl.get_loaders()
    args.audio_dim, args.text_dim, args.video_dim = train_loaders[0].dataset.get_featdim()
    model = get_models(args).cuda()
    return dl, train_loaders, eval_loaders, test_loaders, model


def save_test1_npz(args, model, test_loader, out_npz):
    probs = run_epoch(args, model, None, test_loader, train=False)
    os.makedirs(os.path.dirname(out_npz) or '.', exist_ok=True)
    np.savez_compressed(out_npz, emo_probs=probs, args=np.array(args, dtype=object))
    c = Counter(probs.argmax(1).tolist())
    print(f'saved {out_npz}  ({len(probs)} preds)')
    for i in range(args.output_dim1):
        print(f'   {idx2emo_mer[i]:10s}: {c[i]:6d}')
    return probs


# --------------------------------------------------------------------------- #
#  pseudo-corpus (candidate-ONLY) -- Tier-1 label source
# --------------------------------------------------------------------------- #
def build_pseudo_corpus(args, bridge_logits_npz, real_label_npz, save_npz):
    """Candidate-only pseudo corpus (no real train mixed in). Guards against
    confirmation bias: tau-adjusted logits (test-prior calibrated) + per-class
    top-fraction selection. test1_corpus copied from the real label file so
    test1 inference still covers the scored pool."""
    d0 = np.load(real_label_npz, allow_pickle=True)
    real_train  = dict(d0['train_corpus'].tolist())
    test1_corpus = dict(d0['test1_corpus'].tolist())

    # train prior (idx order) from the REAL corpus
    prior = np.zeros(len(idx2emo_mer), dtype=np.float64)
    for v in real_train.values():
        prior[emo2idx_mer[v['emo']]] += 1
    prior /= prior.sum()

    cand_csv   = os.path.join(config.DATA_DIR['MER2026'], 'track1_track2_candidate.csv')
    cand_names = func_read_key_from_csv(cand_csv, 'name')
    logits = np.array(np.load(bridge_logits_npz, allow_pickle=True)['emo_probs'].tolist(),
                      dtype=np.float64)
    assert len(logits) == len(cand_names), \
        f'logits ({len(logits)}) != candidates ({len(cand_names)})'

    adj  = logits - args.tau * np.log(prior + 1e-12)[None, :]
    adj -= adj.max(1, keepdims=True)
    probs = np.exp(adj); probs /= probs.sum(1, keepdims=True)
    pseudo = probs.argmax(1); conf = probs.max(1)

    by_class = defaultdict(list)
    for i in range(len(cand_names)):
        if conf[i] >= args.min_conf:
            by_class[int(pseudo[i])].append((conf[i], i))
    train_corpus, added = {}, Counter()
    for c, items in by_class.items():
        items.sort(reverse=True)
        keep = max(1, int(round(len(items) * args.top_frac)))
        for _, idx in items[:keep]:
            name = cand_names[idx]
            if name in real_train:          # never use a name that has a real label
                continue
            train_corpus[name] = {'emo': idx2emo_mer[int(pseudo[idx])]}
            added[int(pseudo[idx])] += 1

    np.savez_compressed(save_npz, train_corpus=train_corpus, test1_corpus=test1_corpus)
    print(f'pseudo corpus -> {save_npz}   ({len(train_corpus)} candidate-only pseudo samples)')
    print(f'  tau={args.tau} top_frac={args.top_frac} min_conf={args.min_conf}')
    for i in range(len(idx2emo_mer)):
        print(f'   {idx2emo_mer[i]:10s}: pseudo-added {added[i]:6d}')
    return save_npz


# --------------------------------------------------------------------------- #
#  modes
# --------------------------------------------------------------------------- #
def mode_bridge(args):
    print('===== MODE: bridge (train v3 on real train, save weights + test1 logits) =====')
    dl, train_loaders, eval_loaders, test_loaders, model = build_model_and_loaders(args)
    inner = model.module if hasattr(model, 'module') else model

    # class-weighted CE from the real train labels of fold 0
    labels = np.concatenate([d[1].numpy() for d in train_loaders[0]]).astype(int)
    cls_loss = CELoss(weight=class_weights_from_labels(labels, args.output_dim1)).cuda()

    optimizer = (optim.AdamW if args.optimizer == 'adamw' else optim.Adam)(
        model.parameters(), lr=args.lr, weight_decay=args.l2)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                 eta_min=args.lr * 0.01) if args.use_lr_scheduler else None

    best_f1, best_state, no_imp = -1.0, None, 0
    for ep in range(args.epochs):
        inner.model.current_epoch = ep
        tr = run_epoch(args, model, cls_loss, train_loaders[0], optimizer, train=True)
        if sched is not None: sched.step()
        ev = run_epoch(args, model, None, eval_loaders[0], train=False)
        el = np.concatenate([d[1].numpy() for d in eval_loaders[0]]).astype(int)
        from sklearn.metrics import f1_score
        f1 = f1_score(el, ev.argmax(1), average='weighted') if len(ev) else 0.0
        print(f'  epoch {ep+1:02d}  loss {tr:.4f}  eval-WAF {f1:.4f}')
        if f1 > best_f1:
            best_f1, no_imp = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in inner.model.state_dict().items()}
        else:
            no_imp += 1
            if args.patience > 0 and no_imp >= args.patience:
                print(f'  early stop at epoch {ep+1}'); break

    inner.model.load_state_dict(best_state)
    torch.save(best_state, args.bridge_ckpt)
    print(f'bridge state_dict -> {args.bridge_ckpt}  (best eval-WAF {best_f1:.4f})')
    save_test1_npz(args, model, test_loaders[0], args.bridge_logits)


def mode_tma(args):
    print('===== MODE: tma (re-init head, adapt on candidate pseudo pool) =====')
    real_label = args.label_npz or config.PATH_TO_LABEL['MER2026']
    tma_npz = os.path.join(args.work_dir, f'tma_pseudo_{args.tag}.npz')
    build_pseudo_corpus(args, args.bridge_logits, real_label, tma_npz)
    config.PATH_TO_LABEL['MER2026'] = tma_npz            # dataloader reads this

    dl, train_loaders, eval_loaders, test_loaders, model = build_model_and_loaders(args)
    inner = model.module if hasattr(model, 'module') else model

    # load bridge weights (same arch, strict), then re-init + freeze
    inner.model.load_state_dict(torch.load(args.bridge_ckpt, map_location='cpu'))
    inner.model = inner.model.cuda()
    m = inner.model
    hidden_dim = args.hidden_dim
    dropout    = args.dropout
    num_heads  = max(1, hidden_dim // 64)

    freeze = list(SPEAKER_MODULES)
    reinit = list(REINIT_MODULES)
    if args.freeze_video:
        freeze.append('video_encoder')
    else:
        reinit.insert(0, 'video_encoder')

    # RE-INIT target-adaptable modules (fresh weights, learn on listener dist.)
    m.listener_pool     = LearnableQueryPooling(hidden_dim, num_heads, dropout)
    m.cross_attn_fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
    m.norm_fusion       = nn.LayerNorm(hidden_dim)
    m.fc_out_1          = nn.Linear(hidden_dim, args.output_dim1)
    m.fc_out_2          = nn.Linear(hidden_dim, args.output_dim2)
    if not args.freeze_video:
        from toolkit.models.memocmt_v3 import LSTMSeqEncoder
        m.video_encoder = LSTMSeqEncoder(args.video_dim, hidden_dim, dropout)
    m.speaker_drop_p = args.tma_speaker_drop
    inner.model = m.cuda()

    # FREEZE source modules
    for n in freeze:
        for p in getattr(m, n).parameters():
            p.requires_grad = False
    trainable = [p for p in m.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    assert n_train > 0, 'no trainable params -- check reinit list'
    print(f'  freeze: {freeze}')
    print(f'  reinit/train: {reinit}  ({n_train} params)  speaker_drop_p={args.tma_speaker_drop}')

    # single loader over ALL pseudo candidates (dataset holds every name; the
    # SubsetRandomSampler only restricts fold-0's view -> use the full dataset)
    full_ds = train_loaders[0].dataset
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=full_ds.collater,
                             pin_memory=True)

    pl = np.load(tma_npz, allow_pickle=True)['train_corpus'].tolist()
    plabels = [emo2idx_mer[v['emo']] for v in pl.values()]
    cls_loss = CELoss(weight=class_weights_from_labels(plabels, args.output_dim1)).cuda()

    optimizer = (optim.AdamW if args.optimizer == 'adamw' else optim.Adam)(
        trainable, lr=args.tma_lr, weight_decay=args.l2)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tma_epochs,
                                                 eta_min=args.tma_lr * 0.01) if args.use_lr_scheduler else None

    for ep in range(args.tma_epochs):
        m.current_epoch = ep
        tr = run_epoch(args, model, cls_loss, full_loader, optimizer, train=True, frozen=freeze)
        if sched is not None: sched.step()
        print(f'  epoch {ep+1:02d}/{args.tma_epochs}  loss {tr:.4f}')

    out_npz = args.out_npz or os.path.join(args.work_dir, f'test1_tma_{args.tag}_{int(time.time())}.npz')
    save_test1_npz(args, model, test_loaders[0], out_npz)
    print('\nnext: tau-sweep this npz, e.g.')
    print(f'  for T in 1.5 2.0 2.5; do python3 submission.py adjust_submission '
          f'--result_npz={out_npz} --save_csv=answer.csv --tau=$T; done')


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['bridge', 'tma'])
    # data / feature (mirror run_ensemble.sh COMMON)
    ap.add_argument('--dataset', default='MER2026')
    ap.add_argument('--feat_type', default='frm_unalign')
    ap.add_argument('--audio_feature', default='wavlm-large-FRA')
    ap.add_argument('--text_feature',  default='chinese-roberta-wwm-ext-large-FRA')
    ap.add_argument('--video_feature', default='clip-vit-large-patch14-FRA')
    ap.add_argument('--model', default='memocmt_v3')
    ap.add_argument('--hyper_path', default='toolkit/hyper-sweep/v3_baseline.yaml')
    ap.add_argument('--label_npz', default=None, help='real 6-way label npz (default: config)')
    # training
    ap.add_argument('--optimizer', default='adamw', choices=['adam', 'adamw'])
    ap.add_argument('--l2', type=float, default=0.00001)
    ap.add_argument('--use_lr_scheduler', action='store_true', default=True)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--debug', action='store_true', default=False)
    # bridge outputs
    ap.add_argument('--bridge_ckpt',   default='./tma_out/bridge_v3.pt')
    ap.add_argument('--bridge_logits', default='./tma_out/bridge_v3_test1.npz',
                    help='bridge mode: where to write; tma mode: pseudo source (may point at a strong 5-fold v3 npz)')
    # tma knobs
    ap.add_argument('--work_dir', default='./tma_out')
    ap.add_argument('--tag', default='t1')
    ap.add_argument('--out_npz', default=None)
    ap.add_argument('--tma_lr', type=float, default=1e-4)
    ap.add_argument('--tma_epochs', type=int, default=20)
    ap.add_argument('--tma_speaker_drop', type=float, default=0.3)
    ap.add_argument('--freeze_video', action='store_true', default=True)
    ap.add_argument('--no_freeze_video', dest='freeze_video', action='store_false')
    # pseudo-corpus guards
    ap.add_argument('--tau', type=float, default=2.5)
    ap.add_argument('--top_frac', type=float, default=0.5)
    ap.add_argument('--min_conf', type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.bridge_ckpt) or '.', exist_ok=True)
    torch.cuda.set_device(int(str(args.gpu).split(',')[0]))
    set_seed(args.seed)

    prep_feat_scale(args)
    config.dataset = args.dataset
    if args.label_npz:
        config.PATH_TO_LABEL['MER2026'] = args.label_npz
    model_config = OmegaConf.load(args.hyper_path)[args.model]
    args = merge_args_config(args, model_config)

    if args.mode == 'bridge':
        mode_bridge(args)
    else:
        mode_tma(args)


if __name__ == '__main__':
    main()
