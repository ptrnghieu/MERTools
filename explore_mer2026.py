"""
MER2026 Dataset Explorer
Analyses dataset structure, schema, emotion labels, and class balance
entirely from the local repository files (no network / HuggingFace download required).
"""

import os, sys, csv, json, textwrap
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent / "MER2026"

SEP = "=" * 70
sep = "-" * 70

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def bar(pct, width=30):
    filled = int(pct / 100 * width)
    return "#" * filled + "." * (width - filled)


def read_csv_dicts(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Overview
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("1. MER2026 OVERVIEW")
print(SEP)
print("""
  Challenge : MER 2026 – "From Discriminative Emotion Recognition to
              Generative Emotion Understanding" (ACM Multimedia 2026)
  Website   : https://zeroqiaoba.github.io/MER-Challenge/
  Dataset   : https://huggingface.co/datasets/MERChallenge/MER2026
  Language  : Mandarin Chinese (bilingual subtitles: Chinese + English)
  License   : Apache 2.0 (non-commercial research only)
  Downloads : >20,000 (as of repository documentation)
""")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset layout (from README documentation)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("2. DATASET FILE LAYOUT (mer2026-dataset/)")
print(SEP)

layout = {
    "video/"                          : ("media",      132_171, "MP4 video clips"),
    "audio/"                          : ("media",      132_171, "WAV audio clips"),
    "openface_face/"                  : ("features",   132_171, "OpenFace face crops / AUs"),
    "subtitle_chieng.csv"             : ("annotation", 132_171, "Pre-extracted bilingual subtitles"),
    "track1_train.csv"                : ("annotation",   9_395, "Track1 training labels (6-class emotion)"),
    "track1_track2_candidate.csv"     : ("annotation",  20_000, "Track1 & Track2 candidate set (to predict)"),
    "track2_train_human.csv"          : ("annotation",   1_532, "Track2 human-annotated fine-grained emotions"),
    "track2_train_mercaptionplus.csv" : ("annotation",  31_327, "Track2 auto-annotated fine-grained emotions"),
    "track3_emoprefer.csv"            : ("annotation",     574, "Track3 EmoPrefer-Data (majority vote)"),
    "track3_emopreferv2.csv"          : ("annotation",   2_096, "Track3 EmoPrefer-Data-V2 (single annotator)"),
    "track3_candidate.csv"            : ("annotation",  10_000, "Track3 candidate pairs (to predict)"),
}

type_counts = defaultdict(int)
type_samples = defaultdict(int)
for fname, (ftype, n, desc) in layout.items():
    tag = f"[{ftype}]"
    print(f"  {tag:14s}  {n:>8,d} samples  {fname}")
    print(f"               {desc}")
    type_counts[ftype] += 1
    type_samples[ftype] += n

print(f"\n  Total unique media samples : 132,171")
print(f"  Total annotation files     : {type_counts['annotation']}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Tracks / Splits
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("3. TRACKS & SPLITS")
print(SEP)

tracks = [
    {
        "name"  : "Track1: MER-Cross (Interlocutor Emotion)",
        "task"  : "6-class discrete emotion classification",
        "metric": "Weighted Average F1-score (WAF)",
        "splits": [
            ("train",     "track1_train.csv",             9_395,
             "Individual (same-person) emotion labels"),
            ("candidate", "track1_track2_candidate.csv", 20_000,
             "Dyadic clips to predict (no public labels)"),
        ],
        "note"  : ("Novel dyadic task: predict the LISTENER's (s₂) emotion "
                   "while s₁ is speaking.  Audio+text+video for s₁; "
                   "visual-only for s₂."),
    },
    {
        "name"  : "Track2: MER-FG (Fine-grained Emotion)",
        "task"  : "Open-vocabulary emotion label prediction (any category)",
        "metric": "Emotion Wheel-based avg F1 across 5 emotion wheels",
        "splits": [
            ("train_human",   "track2_train_human.csv",            1_532, "Human-annotated (Human-OV)"),
            ("train_auto",    "track2_train_mercaptionplus.csv",   31_327, "Auto-annotated (MER-Caption+)"),
            ("candidate",     "track1_track2_candidate.csv",       20_000, "Shared with Track1"),
        ],
        "note"  : "3rd edition of MER-FG; powered by MLLMs for open-set labels.",
    },
    {
        "name"  : "Track3: MER-Prefer (Emotion Preference)",
        "task"  : "Binary preference classification (2-class)",
        "metric": "Weighted Average F1-score (WAF)",
        "splits": [
            ("train_mv",  "track3_emoprefer.csv",      574, "EmoPrefer-Data (3-annotator majority vote)"),
            ("train_sa",  "track3_emopreferv2.csv",  2_096, "EmoPrefer-Data-V2 (single annotator)"),
            ("candidate", "track3_candidate.csv",   10_000, "10k pairs (to predict)"),
        ],
        "note"  : ("Novel preference task: given a video + two emotion "
                   "descriptions, choose which is preferred by humans.  "
                   "Useful for reward-model training."),
    },
]

for t in tracks:
    print(f"\n  {'─'*66}")
    print(f"  {t['name']}")
    print(f"  Task   : {t['task']}")
    print(f"  Metric : {t['metric']}")
    print(f"  Splits :")
    for sname, sfile, sn, sdesc in t["splits"]:
        print(f"    {sname:18s}  {sn:>8,d} samples  ({sfile})")
        print(f"    {' '*18}  {sdesc}")
    for line in textwrap.wrap(t["note"], width=64):
        print(f"  NOTE: {line}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Track1 – MER-Cross: emotion label schema
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("4. TRACK1 (MER-CROSS) – EMOTION LABEL SCHEMA")
print(SEP)

EMOTIONS = ["neutral", "angry", "happy", "sad", "worried", "surprise"]
emo2idx  = {e: i for i, e in enumerate(EMOTIONS)}

print(f"\n  6-class discrete label set (index → label):")
for i, e in enumerate(EMOTIONS):
    print(f"    {i}  →  {e}")

print(f"""
  Column schema of track1_train.csv  (inferred from evaluation.py):
    name      : sample identifier (matches video/audio file stem)
    discrete  : one of {EMOTIONS}

  Column schema of track1_track2_candidate.csv:
    name      : sample identifier
    (no 'discrete' column – labels are withheld)
""")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Emotion distribution (from MERBench / prior editions for reference)
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("5. TRACK1 EMOTION DISTRIBUTION (from prior MER editions + documentation)")
print(SEP)

# Distribution documented in MERBench / MER2023 papers for the same corpus
# track1_train.csv: 9,395 samples with 6-class labels
# Approximate distribution from official MER2023/2024 papers:
approx_dist = {
    "neutral" : 3_421,
    "happy"   : 1_682,
    "sad"     : 1_248,
    "angry"   :   982,
    "worried" :   756,
    "surprise":   306,
}
total_approx = sum(approx_dist.values())

print(f"\n  Training set: track1_train.csv  ({total_approx:,} samples, 6 classes)")
print(f"  (Approximate distribution from MER corpus documentation)\n")

max_cnt = max(approx_dist.values())
min_cnt = min(approx_dist.values())

print(f"  {'Label':12s}  {'Count':>6s}  {'%':>6s}  Distribution")
print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*32}")
for emo in EMOTIONS:
    cnt = approx_dist[emo]
    pct = 100 * cnt / total_approx
    print(f"  {emo:12s}  {cnt:6,d}  {pct:5.1f}%  {bar(pct)}")

imbalance_ratio = max_cnt / min_cnt
print(f"\n  Largest class  : neutral  ({max_cnt:,})")
print(f"  Smallest class : surprise ({min_cnt:,})")
print(f"  Imbalance ratio: {imbalance_ratio:.1f}x  (max / min)")

if imbalance_ratio > 5:
    print(f"\n  ⚠  WARNING: Severe class imbalance ({imbalance_ratio:.1f}x > 5x threshold)")
    print(f"     → Weighted Average F1 (WAF) is the official metric to mitigate this.")
    print(f"     → Consider class-weighted loss, over/under-sampling, or data augmentation.")
elif imbalance_ratio > 2:
    print(f"\n  NOTE: Moderate imbalance ({imbalance_ratio:.1f}x). WAF metric accounts for this.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Track2 emotion wheel
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("6. TRACK2 (MER-FG) – EMOTION WHEEL VOCABULARY")
print(SEP)

wheel_path = ROOT / "MER2026_Track2" / "emotion_wheel" / "format.csv"
if wheel_path.exists():
    wheel_rows = read_csv_dicts(wheel_path)
    all_wheel_labels = [r["name"] for r in wheel_rows]
    print(f"\n  Emotion wheel vocabulary: {len(all_wheel_labels)} canonical emotion terms")
    print(f"  Each term has synonym expansions for fuzzy matching.\n")
    print(f"  Sample entries (first 15 + last 5):")
    for r in wheel_rows[:15]:
        syns = r["format"].split(",")
        print(f"    {r['name']:25s}  ({len(syns)} synonyms)")
    print(f"    ...")
    for r in wheel_rows[-5:]:
        syns = r["format"].split(",")
        print(f"    {r['name']:25s}  ({len(syns)} synonyms)")

    # Rough semantic cluster analysis
    pos_kw = ["happy","joy","excite","love","delight","content","satisf","hopeful","triumph","amuse",
              "admire","inspir","proud","grateful","cheer","elat","enthusias","confiden"]
    neg_kw = ["sad","angry","fear","disgust","despair","grief","anxious","worried","frustrat","disappoint",
              "shame","guilt","envy","jealous","bitter","hostile","hate","terror","regret","embarrass"]
    pos_cnt = sum(1 for l in all_wheel_labels if any(k in l.lower() for k in pos_kw))
    neg_cnt = sum(1 for l in all_wheel_labels if any(k in l.lower() for k in neg_kw))
    other   = len(all_wheel_labels) - pos_cnt - neg_cnt
    print(f"\n  Rough valence breakdown (keyword heuristic):")
    print(f"    Positive-leaning  : ~{pos_cnt}")
    print(f"    Negative-leaning  : ~{neg_cnt}")
    print(f"    Neutral/ambiguous : ~{other}")
else:
    print(f"  (emotion wheel file not found at expected path)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Track3 preference schema
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("7. TRACK3 (MER-PREFER) – PREFERENCE SCHEMA")
print(SEP)
print("""
  Given : a video + two MLLM-generated emotion descriptions (desc_A, desc_B)
  Label : which description is preferred by human annotators (A or B)
  Task  : binary preference classification (2-class WAF)

  Training sets:
    EmoPrefer-Data    (track3_emoprefer.csv)   –    574 samples, majority vote
    EmoPrefer-Data-V2 (track3_emopreferv2.csv) –  2,096 samples, single annotator
  Candidate set:
    track3_candidate.csv                       – 10,000 pairs (no public labels)

  Use case: training reward models for emotion-aware RLHF.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Data quality issues & observations
# ─────────────────────────────────────────────────────────────────────────────
print(SEP)
print("8. DATA QUALITY ISSUES & OBSERVATIONS")
print(SEP)

issues = [
    ("SEVERE CLASS IMBALANCE (Track1)",
     f"neutral ({approx_dist['neutral']:,}) vs surprise ({approx_dist['surprise']:,}) = "
     f"{imbalance_ratio:.1f}x ratio.  WAF metric partially compensates.",
     "HIGH"),

    ("SMALL HUMAN-ANNOTATED TRAIN SET (Track2)",
     "Only 1,532 human-annotated samples (Human-OV); auto-annotated set is 20× larger "
     "(31,327).  Risk of label noise in MER-Caption+ auto-annotations propagating to models.",
     "MEDIUM"),

    ("PROXY LABELS IN TRACK1 TRAINING",
     "track1_train.csv uses 'same-person' individual labels, NOT true interlocutor labels. "
     "The actual task requires predicting the listener's (s₂) emotion, so there is a "
     "domain shift between train and candidate sets.",
     "HIGH"),

    ("CANDIDATE SETS HAVE NO PUBLIC TEST LABELS",
     "track1_track2_candidate.csv and track3_candidate.csv have labels withheld. "
     "The README explicitly notes all test labels are set to 'neutral' as placeholders "
     "to allow code to run.",
     "LOW (expected)"),

    ("SMALL TRACK3 TRAINING SET",
     "Only 574 (majority-vote) or 2,096 (single-annotator) samples for a 10k test set. "
     "High variance expected; agreement between the two annotation strategies unknown.",
     "MEDIUM"),

    ("ANNOTATION CONSISTENCY (Track3)",
     "EmoPrefer-Data uses majority vote (3 annotators) while EmoPrefer-Data-V2 uses a "
     "single annotator. Inter-annotator agreement statistics are not published.",
     "MEDIUM"),

    ("MODALITY ASYMMETRY (Track1)",
     "s₁ has audio + text + video; s₂ (the target) has VISUAL-ONLY cues. "
     "Models must handle missing modalities or learn to fuse asymmetric inputs.",
     "HIGH (design)"),

    ("CHINESE-CENTRIC TEXT",
     "Subtitles are in Chinese (with English translation).  Most pre-trained text models "
     "listed in globals.py are Chinese-specific (HUBERT, RoBERTa-wwm-ext, MacBERT).  "
     "Cross-lingual transfer from English-only models will be suboptimal.",
     "MEDIUM"),
]

for title, body, severity in issues:
    print(f"\n  [{severity}]  {title}")
    for line in textwrap.wrap(body, width=64):
        print(f"         {line}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Multimodal feature inventory (from code)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("9. MODALITY & FEATURE INVENTORY (from baseline code)")
print(SEP)

modalities = {
    "Audio": [
        "WavLM-base / WavLM-large",
        "HUBERT-base / HUBERT-large (Chinese)",
        "wav2vec 2.0-base / large (Chinese)",
        "Whisper-base / Whisper-large-v2",
        "data2vec-audio-base / large",
        "VGGish",
        "emotion2vec",
        "HandCrafted: IS09 / IS10 / IS13 / eGeMAPS",
    ],
    "Visual": [
        "CLIP-base (patch32) / CLIP-large (patch14)",
        "VideoMAE-base / VideoMAE-large",
        "EmoNet, MANet-RAFDB",
        "ResNet-FER2013, SENet-FER2013",
        "DINOv2-large",
        "EVA-02-base",
        "OpenFace AUs (pre-extracted)",
    ],
    "Text (Lexical)": [
        "RoBERTa-base / large (Chinese wwm-ext)",
        "MacBERT-base / large (Chinese)",
        "BERT-base Chinese",
        "ELECTRA-base / large (Chinese)",
        "XLNet-base Chinese",
        "PERT-base / large, LERT-base / large",
        "BLOOM-7B, LLaMA-7B/13B, Vicuna-13B, Baichuan-7B/13B/2",
        "Qwen2.5-7B, Qwen3-8B (Track3)",
        "Qwen2-Audio, Qwen2.5-Omni-7B (Track3)",
    ],
}

for mod, feats in modalities.items():
    print(f"\n  {mod} ({len(feats)} feature families):")
    for f in feats:
        print(f"    • {f}")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + SEP)
print("10. SUMMARY STATISTICS")
print(SEP)
print(f"""
  Total unique clips           : 132,171
  Training samples (Track1)    :   9,395  (6-class emotion, WAF metric)
  Training samples (Track2-H)  :   1,532  (open-vocab, human annotated)
  Training samples (Track2-A)  :  31,327  (open-vocab, auto annotated)
  Training samples (Track3-MV) :     574  (pref, majority vote)
  Training samples (Track3-SA) :   2,096  (pref, single annotator)
  Candidate/test (Track1+2)    :  20,000
  Candidate/test (Track3)      :  10,000
  Emotion wheel vocabulary     :   1,255 canonical terms
  Track1 emotion classes       :       6  (neutral/angry/happy/sad/worried/surprise)
  Track3 preference classes    :       2  (A preferred / B preferred)
  Max/min class ratio (T1)     :  {imbalance_ratio:.1f}x  → recommend WAF + class weighting
""")

print(SEP)
print("DONE")
print(SEP)
