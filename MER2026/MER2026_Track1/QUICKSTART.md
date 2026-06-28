# MER2026 Track 1 — Quickstart Guide

Hướng dẫn từng bước để chạy training và tạo file submission cho **MER-Cross (Track 1)** sử dụng pre-extracted features tại HuggingFace: [`hhieupt/mer2026-features`](https://huggingface.co/datasets/hhieupt/mer2026-features).

---

## Mục lục

1. [Mô tả bài toán và dữ liệu](#1-mô-tả-bài-toán-và-dữ-liệu)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Clone repo ban tổ chức](#3-clone-repo-ban-tổ-chức)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Tải dữ liệu từ HuggingFace](#5-tải-dữ-liệu-từ-huggingface)
6. [Giải nén và tổ chức thư mục](#6-giải-nén-và-tổ-chức-thư-mục)
7. [Tạo file config.py](#7-tạo-file-configpy)
8. [Training](#8-training)
9. [Tạo file submission](#9-tạo-file-submission)
10. [Lưu ý và mẹo](#10-lưu-ý-và-mẹo)

---

## 1. Mô tả bài toán và dữ liệu

### Bài toán MER-Cross

MER-Cross là task nhận dạng cảm xúc trong hội thoại song thoại (dyadic conversation). Trong mỗi lượt nói:

- **s₁** (speaker): người đang nói — có audio, text.
- **s₂** (listener): người đang nghe — chỉ có video (biểu cảm khuôn mặt)

**Mục tiêu**: dự đoán cảm xúc của **s₂ (listener)** dựa trên audio + text của s₁ và video của s₂.

### Nhãn cảm xúc (6 lớp)

| Index | Nhãn | Ý nghĩa |
|---|---|---|
| 0 | `neutral` | Trung tính |
| 1 | `angry` | Tức giận |
| 2 | `happy` | Vui vẻ |
| 3 | `sad` | Buồn |
| 4 | `worried` | Lo lắng |
| 5 | `surprise` | Ngạc nhiên |

### Dữ liệu

| Tập | Số mẫu | Nhãn |
|---|---|---|
| Train | 9,395 | Có (6 lớp) |
| Test | 20,000 | Không (dự đoán và submit) |

Train trên Individual data (1 người - có đầy đủ 3 modalities)
Test trên Interlocutor data ( 2 người: speaker có audio+text, listener có visual)

### Features có sẵn trong repo `hhieupt/mer2026-features`

Repo cung cấp 2 loại features đã được trích xuất sẵn:

**Frame-level features (FRA)** — giữ nguyên thông tin theo thời gian:

| File | Modality | Model trích xuất | Chiều | Kích thước |
|---|---|---|---|---|
| `wavlm-large-FRA.zip` | Audio | WavLM-Large | 1024 | 18.9 GB |
| `chinese-roberta-wwm-ext-large-FRA.zip` | Text | Chinese RoBERTa-wwm-ext-Large | 1024 | 5.96 GB |
| `clip-vit-large-patch14-FRA.zip` | Video | CLIP-ViT-Large-Patch14 | 768 | 5 GB |

**Utterance-level features (UTT)** — mỗi mẫu là một vector duy nhất:

| Nội dung | Models | Kích thước |
|---|---|---|
| `features.zip` (audio + text + video) | Chinese HuBERT-Large, Chinese MacBERT-Large, CLIP-ViT-Large,... | 1.12 GB |

**File nhãn và metadata:**

| File | Mô tả |
|---|---|
| `track1_label_6way.npz` | File nhãn đã xử lý cho training code |
| `track1_train.csv` | Danh sách + nhãn 9,395 mẫu train |
| `track1_track2_candidate.csv` | Danh sách 20,000 mẫu test cần dự đoán gồm cả cho track 1 và track 2 |

---

## 2. Yêu cầu hệ thống

- **OS**: Linux (khuyến nghị Ubuntu 20.04+)
- **Python**: 3.8 trở lên
- **GPU**: CUDA-compatible, khuyến nghị ≥ 16GB VRAM
- **RAM**: ≥ 32GB (features được load toàn bộ vào RAM khi training)
- **Dung lượng ổ đĩa**: ≥ 40GB cho toàn bộ features

> Nếu không có GPU, training vẫn chạy được trên CPU nhưng rất chậm.

---

## 3. Clone repo ban tổ chức

```bash
git clone https://github.com/zeroQiaoba/MERTools.git
cd MERTools/MER2026/MER2026_Track1
```

> Tất cả các lệnh trong hướng dẫn này đều chạy từ thư mục `MER2026_Track1/`.

---

## 4. Cài đặt môi trường

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scikit-learn omegaconf pandas tqdm matplotlib openai fire huggingface_hub
pip install opencv-python-headless pytorchvideo ftfy timm einops decord regex iopath
```

> Thay `cu118` bằng phiên bản CUDA phù hợp với máy bạn (cu121, cu124, v.v.). Kiểm tra phiên bản CUDA bằng `nvidia-smi`.

---

## 5. Tải dữ liệu từ HuggingFace

Đặt đường dẫn lưu dữ liệu (thay đổi theo máy của bạn):

```bash
HF_DIR=/path/to/download/mer2026-hf   # thư mục tải về từ HuggingFace
```

Tải toàn bộ repo:

```bash
python3 - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="hhieupt/mer2026-features",
    repo_type="dataset",
    local_dir="$HF_DIR",
    # token="YOUR_HF_TOKEN",  # bỏ comment nếu repo private
)
print("Download complete!")
EOF
```

Hoặc tải từng file riêng nếu không cần tất cả:

```bash
python3 - <<EOF
from huggingface_hub import hf_hub_download

files = [
    "track1_label_6way.npz",
    "track1_track2_candidate.csv",
    "features.zip",                          # UTT features (~1.12 GB)
    # "wavlm-large-FRA.zip",                 # FRA audio (~18.9 GB)
    # "chinese-roberta-wwm-ext-large-FRA.zip", # FRA text (~5.96 GB)
    # "clip-vit-large-patch14-FRA.zip",      # FRA video (~5 GB)
]

for f in files:
    path = hf_hub_download(
        repo_id="hhieupt/mer2026-features",
        filename=f,
        repo_type="dataset",
        local_dir="$HF_DIR",
    )
    print(f"Downloaded: {path}")
EOF
```

---

## 6. Giải nén và tổ chức thư mục

Đặt đường dẫn thư mục dữ liệu chính (thay đổi theo máy của bạn):

```bash
DATA_DIR=/path/to/your/mer2026   # thư mục chứa dữ liệu đã xử lý
HF_DIR=/path/to/download/mer2026-hf
mkdir -p $DATA_DIR/embeddings
```

Copy file nhãn và metadata:

```bash
cp $HF_DIR/track1_label_6way.npz        $DATA_DIR/
cp $HF_DIR/track1_track2_candidate.csv  $DATA_DIR/
```

Giải nén features (chọn loại bạn cần):

```bash
# UTT features (nhỏ, nhanh, phù hợp để bắt đầu)
unzip $HF_DIR/features.zip -d $DATA_DIR/embeddings/

# FRA features (lớn hơn, nhiều thông tin hơn)
unzip $HF_DIR/wavlm-large-FRA.zip                    -d $DATA_DIR/embeddings/
unzip $HF_DIR/chinese-roberta-wwm-ext-large-FRA.zip  -d $DATA_DIR/embeddings/
unzip $HF_DIR/clip-vit-large-patch14-FRA.zip         -d $DATA_DIR/embeddings/
```

Kiểm tra cấu trúc thư mục sau khi giải nén:

```
$DATA_DIR/embeddings/
├── features/
│   ├── chinese-hubert-large-UTT/           # audio UTT (từ features.zip)
│   ├── chinese-macbert-large-UTT/          # text UTT (từ features.zip)
│   ├── clip-vit-large-patch14-UTT/         # video UTT (từ features.zip)
├── wavlm-large-FRA/                    # audio FRA
├── chinese-roberta-wwm-ext-large-FRA/  # text FRA
├── clip-vit-large-patch14-FRA/         # video FRA
├── track1_label_6way.npz
└── track1_track2_candidate.csv
```

Mỗi thư mục feature chứa các file `.npy`, một file cho mỗi sample:

```
chinese-hubert-large-UTT/
├── sample_00001.npy   # shape: (1024,) hoặc (1, 1024)
├── sample_00002.npy
└── ...
```

---

## 7. Tạo file config.py

Tạo file `config.py` trong thư mục `MER2026_Track1/` (cùng cấp với `main-release.py`). Thay `DATA_DIR` bằng đường dẫn thực tế của bạn:

```python
import os

PATH_TO_LABEL              = {'MER2026': os.path.join(DATA_DIR['MER2026'], 'track1_label_6way.npz')}
PATH_TO_FEATURES           = {'MER2026': os.path.join(DATA_DIR['MER2026'], 'embeddings/features')} (bỏ features nếu là frame-level embeddings)
```

## 8. Training

### Giải thích tham số

| Tham số | Ý nghĩa |
|---|---|
| `--model` | Kiến trúc model:`attention` (baseline đơn giản) |
| `--feat_type` | `utt` = utterance-level (1 vector/mẫu), `frm_unalign` = frame-level chưa căn chỉnh |
| `--audio_feature` | Tên thư mục chứa audio features |
| `--text_feature` | Tên thư mục chứa text features |
| `--video_feature` | Tên thư mục chứa video features |
| `--epochs` | Số epoch training mỗi fold |
| `--gpu` | GPU ID (0, 1, 2, ...). Nếu chỉ có 1 GPU thì dùng `--gpu=0` |
| `--save_root` | Thư mục lưu kết quả training |

### Phương án A — UTT features (khuyến nghị để bắt đầu)

Nhanh hơn, ổn định hơn, phù hợp để kiểm tra pipeline:

```bash
python main-release.py \
  --dataset=MER2026 \
  --model=attention \
  --feat_type=utt \
  --audio_feature=chinese-hubert-large-UTT \
  --text_feature=chinese-macbert-large-UTT \
  --video_feature=clip-vit-large-patch14-UTT \
  --epochs=100 \
  --gpu=0 \
  --save_root=./saved-utt \
  > train_utt.log 2>&1 &

echo "Training started, PID: $!"
tail -f train_utt.log
```

### Phương án B — FRA features (frame-level)

Giữ thông tin temporal:

```bash
python main-release.py \
  --dataset=MER2026 \
  --model=attention \
  --feat_type=frm_unalign \
  --audio_feature=wavlm-large-FRA \
  --text_feature=chinese-roberta-wwm-ext-large-FRA \
  --video_feature=clip-vit-large-patch14-FRA \
  --epochs=100 \
  --gpu=0 \
  --save_root=./saved-fra \
  > train_fra.log 2>&1 &

echo "Training started, PID: $!"
tail -f train_fra.log
```

### Theo dõi tiến trình

Mỗi epoch in ra:
```
epoch:1; metric:emo; train results:0.5628; eval results:0.7122
epoch:2; metric:emo; train results:0.6691; eval results:0.7369
...
```

- `train results`: WAF (Weighted Average F1) trên tập train của fold hiện tại
- `eval results`: WAF trên tập validation của fold hiện tại

Training chạy **5-fold cross-validation** kết quả được lưu vào:

```
./saved-utt-trimodal/result/
├── cv_..._f1:0.XXXX_acc:0.XXXX_....npz     # CV results (tham khảo)
└── test1_..._f1:0.XXXX_acc:0.XXXX_....npz  # Test predictions (dùng để submit)
```

> Chỉ số `f1` trong tên file `test1_*` **không phải** WAF thật trên test set vì test không có nhãn thật — đây là số tính trên nhãn giả (`neutral`). WAF thật chỉ biết sau khi submit lên Codabench.

---

## 9. Tạo file submission

Tìm file predictions của test set:

```bash
ls ./saved-utt-trimodal/result/test1_*.npz
```

Tạo `answer.csv` bằng script có sẵn:

```bash
NPZ=$(ls ./saved-utt-trimodal/result/test1_*.npz | head -1)
echo "Using: $NPZ"

python submission.py generate_submission \
  --result_npz="$NPZ" \
  --save_csv=answer.csv
```

> **Lưu ý**: `submission.py` đọc thứ tự tên mẫu từ `track1_track2_candidate.csv` (qua `config.py`). Không dùng script tự viết để tránh lệch thứ tự predictions.

Kiểm tra file:

```bash
head -5 answer.csv
# name,discrete
# sample_00001,neutral
# sample_00002,happy
# ...

wc -l answer.csv   # phải là 20001 (20000 mẫu + 1 dòng header)
```

Nén và submit:

```bash
zip answer.zip answer.csv
```

Upload file `answer.zip` lên [Codabench](https://www.codabench.org/competitions/17195/) tại trang submit của MER2026 Track 1.

---


**Về lỗi thường gặp:**
- `ModuleNotFoundError`: cài thêm package bị thiếu bằng `pip install <package>`
- `FileNotFoundError` khi đọc features: kiểm tra `PATH_TO_FEATURES` trong `config.py` và tên thư mục feature có khớp với `--audio_feature`, `--text_feature`, `--video_feature` không
- `CUDA out of memory`: giảm `--batch_size` (mặc định là 32)
