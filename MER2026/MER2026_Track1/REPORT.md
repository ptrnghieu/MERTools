# Báo cáo tổng quát — MER2026 Track-1 (MER-Cross)

**Nhiệm vụ:** dự đoán cảm xúc người **nghe** (listener) trong hội thoại tay đôi.
**Kết quả tốt nhất:** **WAF 65.80** (ensemble v3 multi-seed) / 65.78 (v3 đơn).
**Mục tiêu 70: không đạt** — xác nhận trần thông tin ~66 bền vững qua toàn bộ đòn bẩy có cơ sở.

---

## Tóm tắt điều hành

Sau khi khai thác hệ thống các trục — kiến trúc (v1–v11), đặc trưng (CLIP/AU/FER),
chiến lược huấn luyện, ensemble, transductive self-training, và MLLM
(fine-tune) — mô hình tốt nhất giữ nguyên ở **~66 WAF**. Kết luận trung tâm:
nút thắt **không** nằm ở kiến trúc, encoder hay cách huấn luyện, mà ở **giới hạn
thông tin cố hữu** của việc suy cảm xúc người nghe từ khuôn mặt trong bối cảnh
cross-role (train ≠ test về vai). Đây là một nghiên cứu với nhiều **kết quả âm
có giá trị chẩn đoán**, khoanh vùng rõ đâu là rào cản thật.

---

## 1. Định nghĩa bài toán

| Khía cạnh | Chi tiết |
|---|---|
| Nhiệm vụ | Đoán cảm xúc **listener** (s₂) trong đối thoại 2 người |
| Train | 9,395 mẫu **Individual** — audio/text/video **cùng 1 người (đang nói)** |
| Test (chấm) | 574 mẫu **Interlocutor** — video = listener (im lặng), audio/text = **speaker (người khác)** |
| Candidate pool | 20,000 mẫu Interlocutor **không nhãn** (transductive) |
| Nhãn | 6 lớp: neutral / angry / happy / sad / worried / surprise |
| Metric | WAF (weighted-F1) trên Codabench, **nhiễu ±3–4** (chỉ 574 mẫu chấm) |

---

## 2. Thách thức cốt lõi: Cross-role domain shift

Train (speaker-centric) và test (listener-centric) **lệch vai**. Ba phát hiện
nền tảng chi phối toàn bộ dự án:

1. **Video → cảm xúc: TRANSFER.** Quan hệ mặt→cảm xúc (FER) giữ nguyên bất kể
   vai. Là modality đáng tin duy nhất khi chuyển miền.
2. **Audio/text → cảm xúc: KHÔNG transfer sạch** (lời speaker ≠ cảm xúc listener),
   nhưng mang **contagion yếu (~+7 WAF)** so với video-only → vẫn có giá trị.
3. **In-domain CV KHÔNG dự báo test.** Eval trên train (cùng miền) đạt ~0.77–0.87
   nhưng test (khác miền) chỉ ~0.50–0.66. Chỉ Codabench WAF là thước đo thật.

---

## 3. Mô hình tốt nhất: MemoCMT v3 (65.78)

**v1 (gốc):** nhánh speaker (audio↔text cross-attention 2 chiều → gộp) + nhánh
listener (video → LearnableQueryPooling) + fusion (Q=listener, K/V=speaker).

**v3 = v1 + 2 sửa lỗi:**
- **Fix2:** pool a2t và t2a **RIÊNG** (`sp_a`, `sp_t`) → audio/text cân bằng bất
  kể số token khác nhau.
- **Fix3:** fusion K/V = **2 token** `[sp_a, sp_t]` → attention không suy biến.

**Bổ trợ giữ lại (đều +):** inverse-freq class-weight; label-shift hậu kỳ
`logit − τ·log(train_prior)`, **τ=2.5**.

*(Sơ đồ kiến trúc chi tiết mọi biến thể: xem `EXPERIMENTS_ARCHITECTURES.md`.)*

---

## 4. Toàn bộ thí nghiệm

### 4.1 Kiến trúc (feature-fusion)

| Model | Ý tưởng | WAF |
|---|---|---|
| **v3** | Fix2 + Fix3 | **65.78 ✅** |
| v1 | MemoCMT gốc | nền |
| v2 | video-anchor (BiLSTM + gated) — tăng capacity video | 63–64 ✗ |
| v4 | fusion attend full-seq, trộn 1:1 | ~64 ✗ |
| v5 | v4 + gate học per-channel | ~64 ✗ |
| v6 | v3 + CM-StEW aux (dịch video→speaker + align) | 65.36 ~ |
| v7 | v3 + Nonverbal Conflict Exposure (tráo speaker khác-nhãn) | 62–63 ✗ |
| v8 | VISAFF reliability-gated video-anchor | 63 ✗ |
| v9 | 2 nhánh video (CLIP + AU) gated embedding | 60–61 ✗ |
| v10 | v3 + AU làm token fusion thứ 3 | 60–61 ✗ |
| v11 | v3 + aux CE video-only (deep supervision) | ~65 ~ |

→ **Ba trục nhánh video (capacity v2 / information v9,v10 / training-signal
v6,v11) đều âm hoặc trung tính.** Bottleneck không ở kiến trúc.

### 4.2 Đặc trưng, huấn luyện, hậu kỳ

| Thử | Kết quả |
|---|---|
| AU/landmark/pose (py-feat trên crop) | −5 ✗ (observation-shift: speaker nói ↔ listener im lặng) |
| class-weight (inverse-freq) | giữ, + |
| label-shift τ=2.5 | giữ, + |
| **FER backbone** (HSEmotion enet_b2_8, AffectNet-8, THAY CLIP) | **59.6–61 ✗** (in-domain 0.77) |

### 4.3 Transductive (20k candidate không nhãn)

| Thử | Kết quả |
|---|---|
| Pseudo-label trộn vào train, retrain full | 66.0 (+0.2) ~ |
| **TMA/BBA** (bridge v3 → re-init head → train thuần pseudo) | **61.47 (−4.3) ✗** |

### 4.4 Ensemble

| Cấu hình | WAF |
|---|---|
| **v3 × 7 (multi-seed/hyperparam)** | **65.80** |
| all-13 (v3+v6+v11) tau=2.5 | 65.04 |
| all-13 tau=2.0 | 64.76 |

→ **Flat.** Các model quá tương quan (cùng arch, cùng feature) → trung bình không
decorrelate lỗi. v6/v11 kéo xuống.

### 4.5 MLLM (Qwen2.5-VL-7B, QLoRA fine-tune)

| Biến thể | in-domain WAF | test WAF |
|---|---|---|
| Zero-shot (gemini/gpt-4o, prompt FACS) | — | ~42 |
| Direct-SFT + transcript | 0.867 | **49.84** |
| Video-only (bỏ transcript) | 0.594 | **42.8** |
| + MLLS calibrate | — | 43.46 (tệ hơn) |

→ Xa 65.78. Chi tiết vì sao ở §6.

---

## 5. Phát hiện chính (insights)

1. **Chỉ video transfer; audio/text là contagion yếu.** Xác nhận định lượng: bỏ
   transcript khỏi MLLM làm test tụt 49.84→42.8 (≈ −7 = đúng giá trị contagion đã
   biết từ feature-fusion).
2. **Trần thông tin ~66**, bền qua **mọi** trục. Không phải giới hạn kiến trúc.
3. **In-domain ≠ transfer.** MLLM direct-SFT: 0.867 in-domain nhưng 0.498 test —
   khoảng cách −37 do model bám **shortcut** (train transcript = chính người
   trong ảnh → proxy hoàn hảo in-domain, sụp khi test đổi vai).
4. **Vision generic > vision chuyên FER cho task này.** CLIP (65.78) > HSEmotion
   AffectNet (59.6). FER model bị **domain gap posed↔conversational** tệ hơn CLIP.
5. **Ensemble cần đa dạng thật.** Multi-seed cùng arch/feature → tương quan cao →
   không có gain.

---

## 6. Phân tích các thất bại lớn (chẩn đoán)

**MLLM (42–50):** không phải bug (in-domain 0.87/0.59 chứng minh pipeline đúng).
Nguyên nhân: (a) **vision yếu** — Qwen ViT generic **bị freeze** đọc mặt **112²
thấp** không bắt được micro-expression tinh của listener (face-only chỉ 0.43
test); (b) sức mạnh reasoning của LLM vô dụng vì nút thắt là **tri giác thị giác
chi tiết**, không phải suy luận.

**FER backbone (59.6–61):** giả thuyết "vision chuyên FER transfer tốt hơn" bị
lật. HSEmotion train trên **AffectNet posed/biểu cảm mạnh, chính diện**; mặt
listener **tinh tế, hội thoại, nghiêng, low-res** → feature FER lệch miền, đọc
kém. CLIP generic mã hoá thông tin rộng hơn → fusion khai thác tốt hơn. Lặp lại
đúng pattern v9/v10 (thêm tín hiệu mặt chuyên biệt → regress 60-61).

**TMA/BBA (61.47):** confirmation-bias trap — distill từ nhãn giả cứng (mất soft
info) = làm nghèo thầy; đồng thời vứt train thật (vốn hữu ích). "Begin anew"
không hợp khi không có tín hiệu cross-role đúng trong train.

**Ensemble (flat):** thiếu decorrelation.

---

## 7. Kết luận

Với các phương pháp khả thi trong khuôn khổ (feature-fusion + backbone
pretrained + MLLM 7B), **WAF ~66 là trần thực tế; 70 không đạt được.** Nút thắt
là **thông tin**, không phải mô hình: cảm xúc người nghe im lặng trong nhiều mẫu
**mơ hồ từ khuôn mặt**, và audio/text (thuộc người khác) chỉ cung cấp contagion
yếu. Không encoder/kiến trúc/chiến lược nào đã thử vượt qua giới hạn này.

**Kết quả nộp cuối cùng: WAF 65.80** (ensemble v3 multi-seed, τ=2.5).

### Đóng góp (kể cả negative results)
- Kiến trúc v3 (Fix2 + Fix3) — baseline mạnh, tái lập được.
- Chẩn đoán định lượng: video transfer, audio/text = contagion, in-domain ≠
  transfer, generic-vision > FER-vision cho cross-role listener.
- Bác bỏ có bằng chứng: MLLM-from-crops, FER-backbone, transductive TMA, ensemble
  đơn-arch — tất cả không phá được trần.

### Hướng tương lai (nếu tiếp tục, ngoài khuôn khổ hiện tại)
- **Unfreeze vision + crop độ phân giải cao** cho MLLM (chi phí lớn, EV không rõ).
- **Dữ liệu train có cấu trúc cross-role/hội thoại** (nếu ban tổ chức cung cấp) —
  hiện train là Individual nên không dạy được quan hệ speaker→listener thật.
- **Ngữ cảnh hội thoại** (COSMIC/DialogueRNN) — bất khả thi với data hiện tại
  (mẫu rời rạc, không link được lượt trước).

---

## Phụ lục: tài sản code

| File | Vai trò |
|---|---|
| `toolkit/models/memocmt_v3.py` | Kiến trúc winner (65.78). **KHÔNG sửa v1.py.** |
| `toolkit/models/memocmt_v2/v4-v11.py` | Các biến thể (đều ≤ v3) |
| `main-release.py` | Train 5-fold CV + test1 inference |
| `submission.py` | `adjust_submission`(τ), `ensemble_submission`, `mlls_submission`(BCTS+EM), `make_pseudo_corpus` |
| `run_ensemble.sh` | 13-model ensemble runner |
| `extract_fer.py` | Trích feature FER (HSEmotion, GPU fp16) |
| `tma_finetune.py` | BBA/TMA (đã dừng, 61.47) |
| `mllm_dataprep.py` / `mllm_finetune (swift)` / `mllm_infer.py` | Pipeline MLLM (đã dừng, 42-50) |
| `PROBLEM_FRAMING.md` / `EXPERIMENTS_ARCHITECTURES.md` | Khung vấn đề + sơ đồ kiến trúc |
