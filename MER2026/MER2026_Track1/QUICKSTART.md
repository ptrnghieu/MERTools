# MER2026 Track 1 — Quickstart Guide

Hướng dẫn này giúp bạn chạy training và tạo file submission cho **MER-Cross (Track 1)** sử dụng pre-extracted features tại HuggingFace: [`hhieupt/mer2026-features`](https://huggingface.co/datasets/hhieupt/mer2026-features).

---

## Yêu cầu

- Python 3.8+
- CUDA GPU (khuyến nghị ≥ 16GB VRAM)
- ~40GB dung lượng ổ đĩa cho features

---

## Bước 1: Clone repo ban tổ chức

```bash
git clone https://github.com/zeroQiaoba/MERTools.git
cd MERTools/MER2026/MER2026_Track1
```

---

## Bước 2: Cài đặt môi trường

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scikit-learn omegaconf pandas tqdm matplotlib openai fire huggingface_hub
```

---

## Bước 3: Tải features từ HuggingFace

Repo `hhieupt/mer2026-features` chứa 2 bộ features:

| File | Loại | Models | Dung lượng |
|---|---|---|---|
| `wavlm-large-FRA.zip` | Frame-level audio | WavLM-Large | 18.9 GB |
| `chinese-roberta-wwm-ext-large-FRA.zip` | Frame-level text | Chinese RoBERTa-Large | 5.96 GB |
| `clip-vit-large-patch14-FRA.zip` | Frame-level video | CLIP-ViT-Large | 5 GB |
| `features.zip` | Utterance-level (A+T+V) | HuBERT-Large, MacBERT-Large, CLIP-Large | 1.12 GB |
| `track1_train.csv` | Training labels | — | 241 kB |
| `track1_track2_candidate.csv` | Test sample list | — | 400 kB |

Tải xuống (cần HuggingFace token nếu repo private):

```bash
pip install huggingface_hub

python3 - <<'EOF'
from huggingface_hub import hf_hub_download, snapshot_download

# Tải tất cả files
snapshot_download(
    repo_id="hhieupt/mer2026-features",
    repo_type="dataset",
    local_dir="/root/data/mer2026-hf",
    token="YOUR_HF_TOKEN",  # bỏ nếu repo public
)
EOF
```

---

## Bước 4: Giải nén features

```bash
DATA_DIR=/root/data/mer2026
mkdir -p $DATA_DIR/features

# FRA features
unzip /root/data/mer2026-hf/wavlm-large-FRA.zip               -d $DATA_DIR/features/
unzip /root/data/mer2026-hf/chinese-roberta-wwm-ext-large-FRA.zip -d $DATA_DIR/features/
unzip /root/data/mer2026-hf/clip-vit-large-patch14-FRA.zip    -d $DATA_DIR/features/

# UTT features
unzip /root/data/mer2026-hf/features.zip                       -d $DATA_DIR/features/

# CSV files
cp /root/data/mer2026-hf/track1_train.csv            $DATA_DIR/
cp /root/data/mer2026-hf/track1_track2_candidate.csv $DATA_DIR/
```

Sau khi giải nén, cấu trúc thư mục:

```
/root/data/mer2026/
├── features/
│   ├── wavlm-large-FRA/                    # (name.npy mỗi sample)
│   ├── chinese-roberta-wwm-ext-large-FRA/
│   ├── clip-vit-large-patch14-FRA/
│   ├── chinese-hubert-large-UTT/           # từ features.zip
│   ├── chinese-macbert-large-UTT/
│   └── clip-vit-large-patch14-UTT/
├── track1_train.csv
└── track1_track2_candidate.csv
```

---

## Bước 5: Tạo file config.py

Tạo file `config.py` trong thư mục `MER2026_Track1/`:

```python
import os

PATH_TO_PRETRAINED_MODELS = '/root/tools'

DATA_DIR = {
    'MER2026': '/root/data/mer2026',
}

PATH_TO_RAW_VIDEO      = {'MER2026': os.path.join(DATA_DIR['MER2026'], 'video')}
PATH_TO_RAW_AUDIO      = {'MER2026': os.path.join(DATA_DIR['MER2026'], 'audio')}
PATH_TO_RAW_FACE       = {'MER2026': os.path.join(DATA_DIR['MER2026'], 'openface_face')}
PATH_TO_TRANSCRIPTIONS = {'MER2026': os.path.join(DATA_DIR['MER2026'], 'subtitle_chieng.csv')}
PATH_TO_LABEL          = {'MER2026': os.path.join(DATA_DIR['MER2026'], 'track1_label_6way.npz')}
PATH_TO_FEATURES       = {'MER2026': os.path.join(DATA_DIR['MER2026'], 'features')}
```

---

## Bước 6: Generate file nhãn

Tạo `track1_label_6way.npz` từ các file CSV:

```bash
python3 - <<'EOF'
import numpy as np, csv, os

DATA_DIR = '/root/data/mer2026'

def read_csv(path, name_col, label_col=None):
    names, labels = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            names.append(row[name_col])
            labels.append(row.get(label_col, 'neutral') if label_col else 'neutral')
    return names, labels

train_names, train_emos = read_csv(f'{DATA_DIR}/track1_train.csv', 'name', 'discrete')
test_names,  _          = read_csv(f'{DATA_DIR}/track1_track2_candidate.csv', 'name')
test_emos = ['neutral'] * len(test_names)

corpus = {
    'train': {n: {'emo': e} for n, e in zip(train_names, train_emos)},
    'test1': {n: {'emo': e} for n, e in zip(test_names,  test_emos)},
}

save_path = f'{DATA_DIR}/track1_label_6way.npz'
np.savez_compressed(save_path, train_corpus=corpus['train'], test1_corpus=corpus['test1'])
print(f'Saved: {save_path}')
print(f'  Train: {len(train_names)} samples')
print(f'  Test:  {len(test_names)} samples')
EOF
```

---

## Bước 7: Training

### Phương án A — UTT features (nhanh, ổn định)

Dùng features từ `features.zip` với model `cross_role_attention`:

```bash
python main-release.py \
  --dataset=MER2026 \
  --model=cross_role_attention \
  --feat_type=utt \
  --audio_feature=chinese-hubert-large-UTT \
  --text_feature=chinese-macbert-large-UTT \
  --video_feature=clip-vit-large-patch14-UTT \
  --epochs=100 \
  --gpu=0 \
  --save_root=./saved-utt \
  > train_utt.log 2>&1 &
```

### Phương án B — FRA features (frame-level, nhiều thông tin hơn)

Dùng features FRA với `feat_type=frm_unalign`:

```bash
python main-release.py \
  --dataset=MER2026 \
  --model=cross_role_attention \
  --feat_type=frm_unalign \
  --audio_feature=wavlm-large-FRA \
  --text_feature=chinese-roberta-wwm-ext-large-FRA \
  --video_feature=clip-vit-large-patch14-FRA \
  --epochs=100 \
  --gpu=0 \
  --save_root=./saved-fra \
  > train_fra.log 2>&1 &
```

### Theo dõi tiến trình

```bash
tail -f train_utt.log
```

Kết quả mỗi epoch:
```
epoch:10; metric:emo; train results:0.7234; eval results:0.7456
```

Kết thúc training sẽ lưu file kết quả:
```
./saved-utt-trimodal/result/test1_..._f1:0.XXXX_acc:0.XXXX_....npz
```

---

## Bước 8: Tạo file submission

```bash
# Tìm file kết quả test1
NPZ=$(ls ./saved-utt-trimodal/result/test1_*.npz | head -1)
echo "Using: $NPZ"

# Tạo answer.csv
python submission.py generate_submission \
  --result_npz="$NPZ" \
  --save_csv=answer.csv

# Kiểm tra
head -5 answer.csv
wc -l answer.csv   # phải là 20001 (20000 + header)

# Nén và submit
zip answer.zip answer.csv
echo "Ready to submit: answer.zip"
```

Format `answer.csv`:
```
name,discrete
xxxxx,happy
xxxxx,neutral
...
```

Submit file `answer.zip` lên [Codabench](https://www.codabench.org/).

---

## Lưu ý

- **5-fold CV**: Training tự động chạy 5-fold cross-validation, kết quả CV WAF in ra cuối cùng là chỉ số tham khảo trên tập train.
- **Test WAF thật**: Chỉ biết sau khi submit lên Codabench.
- **Nhiều runs**: Chạy nhiều lần với random seed khác nhau rồi average predictions để ổn định hơn.
- **`feat_scale`**: Được set tự động theo `feat_type` (`utt`→1, `frm_align`→6, `frm_unalign`→12), không cần truyền thủ công.
