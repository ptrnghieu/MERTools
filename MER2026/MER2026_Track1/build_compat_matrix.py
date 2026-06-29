"""
Build emotion compatibility matrix from COMET-ATOMIC2020.

Queries oReact relation for each of 6 speaker emotions:
  neutral, angry, happy, sad, worried, surprise

Outputs a 6x6 COMPAT tensor to hardcode into memocmt_fusion.py.

Usage:
    python3 build_compat_matrix.py
    python3 build_compat_matrix.py --threshold 0.10 --topk 20
"""
import argparse
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ── Emotion labels (must match globals.py) ──────────────────────────────────
LABELS = ['neutral', 'angry', 'happy', 'sad', 'worried', 'surprise']
L2I    = {l: i for i, l in enumerate(LABELS)}

# ── Speaker events: "PersonX [displays emotion]" ────────────────────────────
SPEAKER_EVENTS = [
    "PersonX has a neutral expression",        # 0 neutral
    "PersonX expresses anger toward PersonY",  # 1 angry
    "PersonX expresses happiness",             # 2 happy
    "PersonX expresses sadness",               # 3 sad
    "PersonX expresses worry",                 # 4 worried
    "PersonX expresses surprise",              # 5 surprise
]

# ── Keyword → label mapping (covers COMET's common oReact words) ─────────────
WORD_TO_LABEL = {
    # neutral
    'neutral': 0, 'calm': 0, 'indifferent': 0, 'normal': 0,
    'unbothered': 0, 'unaffected': 0, 'okay': 0, 'fine': 0,

    # angry
    'angry': 1, 'anger': 1, 'furious': 1, 'annoyed': 1,
    'irritated': 1, 'mad': 1, 'outraged': 1, 'frustrated': 1,

    # happy
    'happy': 2, 'happiness': 2, 'joy': 2, 'joyful': 2,
    'pleased': 2, 'glad': 2, 'amused': 2, 'delighted': 2,
    'content': 2, 'cheerful': 2, 'elated': 2,

    # sad
    'sad': 3, 'sadness': 3, 'unhappy': 3, 'upset': 3,
    'sorrowful': 3, 'depressed': 3, 'heartbroken': 3,
    'sympathetic': 3, 'empathetic': 3, 'sorry': 3,

    # worried
    'worried': 4, 'scared': 4, 'afraid': 4, 'fear': 4,
    'fearful': 4, 'anxious': 4, 'nervous': 4, 'concerned': 4,
    'uneasy': 4, 'apprehensive': 4, 'threatened': 4,

    # surprise
    'surprised': 5, 'surprise': 5, 'shocked': 5, 'astonished': 5,
    'amazed': 5, 'startled': 5, 'stunned': 5,
}


def words_to_label(text: str):
    """Map a COMET response string → label index, or None if no match."""
    for word in text.lower().split():
        word = word.strip('.,!?')
        if word in WORD_TO_LABEL:
            return WORD_TO_LABEL[word]
    return None


def query_comet(model, tokenizer, event: str, relation: str,
                topk: int, device: str) -> list[str]:
    """
    Query COMET for top-K completions of (event, relation).
    Returns list of decoded strings.
    """
    prompt = f"{event} {relation} [GEN]"
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=topk,
            num_return_sequences=topk,
            max_new_tokens=8,
            early_stopping=True,
        )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def build_matrix(model, tokenizer, topk: int, device: str) -> torch.Tensor:
    """
    Build raw count matrix: counts[speaker_emo][listener_emo].
    """
    counts = torch.zeros(6, 6)

    for spk_idx, event in enumerate(SPEAKER_EVENTS):
        spk_label = LABELS[spk_idx]
        print(f"\n[{spk_label}] event: \"{event}\"")

        responses = query_comet(model, tokenizer, event, 'oReact', topk, device)

        hit_counts = defaultdict(int)
        for resp in responses:
            resp = resp.strip()
            if resp in ('none', 'nothing', 'n/a', ''):
                continue
            lis_idx = words_to_label(resp)
            label_str = LABELS[lis_idx] if lis_idx is not None else '?'
            print(f"   oReact: \"{resp}\"  →  {label_str}")
            if lis_idx is not None:
                counts[spk_idx][lis_idx] += 1
                hit_counts[LABELS[lis_idx]] += 1

        print(f"   hits: {dict(hit_counts)}")

    return counts


def counts_to_compat(counts: torch.Tensor, threshold: float) -> torch.Tensor:
    """Normalize rows → probability, then threshold to 0/1."""
    row_sum = counts.sum(dim=1, keepdim=True).clamp(min=1)
    probs   = counts / row_sum
    compat  = (probs >= threshold).long()
    # Always allow same-emotion pairing (diagonal = 1)
    compat.fill_diagonal_(1)
    return compat, probs


def print_matrix(matrix: torch.Tensor, title: str):
    header = '        ' + '  '.join(f'{l[:3]:>3}' for l in LABELS)
    print(f'\n{title}')
    print(header)
    for i, row in enumerate(matrix):
        vals = '  '.join(f'{v:>3.0f}' if title != 'COMPAT' else f'{int(v):>3}'
                         for v in row)
        print(f'  {LABELS[i][:7]:>7}  {vals}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',     default='mismayil/comet-atomic-2020-bart',
                        help='HuggingFace model id for COMET-ATOMIC2020')
    parser.add_argument('--topk',      type=int,   default=20,
                        help='Number of beam outputs per query')
    parser.add_argument('--threshold', type=float, default=0.10,
                        help='Min probability to mark a pair as compatible')
    parser.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f'Loading COMET model: {args.model}')
    print(f'Device: {args.device}  |  top-k: {args.topk}  |  threshold: {args.threshold}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
    model.eval()

    # ── Query ────────────────────────────────────────────────────────────────
    counts = build_matrix(model, tokenizer, args.topk, args.device)
    compat, probs = counts_to_compat(counts, args.threshold)

    # ── Print results ─────────────────────────────────────────────────────────
    print_matrix(counts, 'RAW COUNTS')
    print_matrix(probs * 100, 'PROBABILITY (%)')
    print_matrix(compat, 'COMPAT')

    # ── Print tensor to paste into memocmt_fusion.py ──────────────────────────
    rows = []
    for i, row in enumerate(compat):
        vals = ', '.join(str(int(v)) for v in row)
        rows.append(f'    [{vals}],  # spk={LABELS[i]}')
    tensor_str = 'torch.tensor([\n' + '\n'.join(rows) + '\n], dtype=torch.long)'

    print('\n' + '='*60)
    print('PASTE INTO memocmt_fusion.py __init__:')
    print('='*60)
    print(f'COMPAT = {tensor_str}')
    print(f'self.register_buffer("compat_matrix", COMPAT)')


if __name__ == '__main__':
    main()
