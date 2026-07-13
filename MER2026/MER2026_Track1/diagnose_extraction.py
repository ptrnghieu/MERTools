"""
diagnose_extraction.py -- check extracted npy counts vs label npz,
and list what train zip files exist in the HF repo.

Run on A100 instance after extraction:
  python diagnose_extraction.py \
    --feat_dir /workspace/feats/clip-vit-large-patch14-336-hires-FRA \
    --label_npz /path/to/track1_label_6way.npz \
    --hf_token $HF_TOKEN \
    --repo MERChallenge/MER2026
"""
import os, sys, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat_dir', required=True)
    ap.add_argument('--label_npz', default='')
    ap.add_argument('--candidate_csv', default='')
    ap.add_argument('--hf_token', default=os.environ.get('HF_TOKEN') or None)
    ap.add_argument('--repo', default='MERChallenge/MER2026')
    ap.add_argument('--list_assets', action='store_true',
                    help='list video_7z/ directory tree in HF repo')
    args = ap.parse_args()

    # --- 1. Count extracted files ---
    all_npy = [f[:-4] for f in os.listdir(args.feat_dir) if f.endswith('.npy')]
    sample_npy  = [n for n in all_npy if n.startswith('sample_')]
    other_npy   = [n for n in all_npy if not n.startswith('sample_')]
    print(f'\n=== Extracted .npy files in {args.feat_dir} ===')
    print(f'  total:     {len(all_npy)}')
    print(f'  sample_*:  {len(sample_npy)}')
    print(f'  other:     {len(other_npy)}')
    if other_npy:
        prefixes = {}
        for n in other_npy:
            pfx = n.split('_')[0] if '_' in n else n[:10]
            prefixes[pfx] = prefixes.get(pfx, 0) + 1
        print(f'  other prefixes: {dict(sorted(prefixes.items(), key=lambda x:-x[1])[:10])}')

    # --- 2. Cross-reference with label npz ---
    if args.label_npz and os.path.exists(args.label_npz):
        data = np.load(args.label_npz, allow_pickle=True)
        print(f'\n=== Label npz keys: {list(data.keys())} ===')
        extracted = set(all_npy)
        for key in ['train_corpus', 'test1_corpus']:
            if key not in data:
                continue
            corpus = data[key].tolist()
            corpus_names = set(corpus.keys())
            found = corpus_names & extracted
            missing = corpus_names - extracted
            extra = extracted - corpus_names  # extracted but not in this split
            print(f'\n  {key}: {len(corpus_names)} names in label')
            print(f'    found in feat_dir:   {len(found)}')
            print(f'    missing from feat:   {len(missing)}')
            if missing and len(missing) <= 20:
                print(f'    missing names: {sorted(missing)[:20]}')
            elif missing:
                print(f'    first 5 missing: {sorted(missing)[:5]}')
    else:
        print('\n(no --label_npz provided, skip label cross-ref)')

    # --- 3. Cross-reference with candidate csv ---
    if args.candidate_csv and os.path.exists(args.candidate_csv):
        import csv
        with open(args.candidate_csv) as f:
            rows = list(csv.DictReader(f))
        cand_names = set(r['name'] for r in rows)
        extracted  = set(all_npy)
        found  = cand_names & extracted
        missing = cand_names - extracted
        print(f'\n=== Candidate CSV ({args.candidate_csv}) ===')
        print(f'  rows: {len(rows)}')
        print(f'  found in feat_dir:  {len(found)}')
        print(f'  missing from feat:  {len(missing)}')
        if missing and len(missing) <= 20:
            print(f'  missing: {sorted(missing)}')

    # --- 4. List HF repo video assets ---
    if args.list_assets:
        try:
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem(token=args.hf_token)
            root = f'datasets/{args.repo}/video_7z'
            print(f'\n=== HF repo: {root} ===')
            for entry in fs.ls(root, detail=True):
                print(f'  {entry["name"]}  ({entry.get("size",0)/1e9:.2f} GB)')
                if entry['type'] == 'directory':
                    for sub in fs.ls(entry['name'], detail=True):
                        print(f'    {sub["name"]}  ({sub.get("size",0)/1e9:.2f} GB)')
        except Exception as e:
            print(f'  list_assets error: {e}')

if __name__ == '__main__':
    main()
