# MER2026 Track-1 (MER-Cross) — Khung tổng thể

> Tổng hợp trạng thái nghiên cứu: định nghĩa bài toán, vấn đề cốt lõi, các cải
> tiến đã thử, blocks, và hướng nghiên cứu đề xuất. Cập nhật tới thí nghiệm TMA/BBA.

---

## 1. Định nghĩa bài toán

| Khía cạnh | Chi tiết |
|---|---|
| **Nhiệm vụ** | Dự đoán cảm xúc người **nghe** (listener, s₂) trong hội thoại tay đôi |
| **Train** | 9,395 mẫu **Individual** — cả 3 modality (audio/text/video) đều của **cùng 1 người đang nói** |
| **Test (chấm điểm)** | 574 mẫu **Interlocutor** — video = listener (im lặng), audio/text = **speaker (người khác)** |
| **Candidate pool** | 20,000 mẫu Interlocutor **không nhãn** (dùng cho transductive) |
| **Nhãn** | 6 lớp: `neutral / angry / happy / sad / worried / surprise` |
| **Metric** | WAF trên Codabench, nhiễu **±3–4** (do chỉ 574 mẫu chấm điểm) |
| **Hiện tại** | **65.78** (v3 + class-weight + label-shift τ=2.5) |
| **Mục tiêu** | **70** (cần +4.2) |

---

## 2. Vấn đề cốt lõi: Cross-role domain shift

Train và test **lệch vai** (speaker-centric ↔ listener-centric). Đây là gốc rễ mọi thứ:

- **Video → cảm xúc: TRANSFER được.** Quan hệ mặt→cảm xúc (FER) giữ nguyên bất kể
  vai. Đây là modality đáng tin duy nhất khi chuyển miền.
- **Audio/text → cảm xúc: KHÔNG transfer sạch.** Lời của speaker ≠ cảm xúc của
  listener. Nhưng mang **contagion yếu** (+7 WAF so với video-only) → vẫn có giá
  trị trong fusion.
- **AU observation-shift:** train = người nói (speech-AU: miệng động), test =
  người nghe im lặng (emotion-AU). py-feat AU trên crop OpenFace → hại −5 WAF.
- **In-domain CV KHÔNG dự báo được test.** Chỉ Codabench WAF mới là thước đo thật.
- **Trần thông tin ~66** — xác nhận qua 15+ thí nghiệm.

---

## 3. Các cải tiến đã thử (theo trục)

| Trục | Thí nghiệm | Kết quả |
|---|---|---|
| **Kiến trúc (winner)** | v1 → **v3**: Fix2 (pool a2t/t2a riêng) + Fix3 (fusion KV = 2 token speaker) | **65.78 ✅** |
| Capacity video | v2 (bidir visual + gated) | 63–64 ✗ |
| Fusion variants | v4/v5 (listener attend full seq / gated mix) | ~64 ✗ |
| Aux loss | v6 (CM-StEW translation+align) | 65.36 ~ |
| Aux loss | v7 (NCE donor-swap) | 62–63 ✗ |
| Aux loss | v8 (VISAFF RGAC) | 63 ✗ |
| Nhánh AU riêng | v9 (CLIP+AU gated embed) | 60–61 ✗ |
| AU token | v10 (AU làm token fusion thứ 3) | 60–61 ✗ |
| Deep-supervision | v11 (aux video-only CE head) | ~65 ~ |
| **Feature** | AU/landmark/pose (py-feat) | −5 ✗ |
| **Training** | class-weight (inverse-freq) | giữ, + |
| **Post-hoc** | label-shift τ=2.5 | giữ, + |
| **Transductive** | Pseudo-label (trộn vào train) | 66.0 (+0.2) ~ |
| **Transductive** | **TMA/BBA (re-init head, train thuần pseudo)** | **61.47 (−4.3) ✗** |
| **LLM** | MLLM zero-shot (gemini/gpt-4o) | WAF ~0.42, cần fine-tune ✗ |

**Ba trục nhánh video (capacity / information / training-signal) đều âm** → bottleneck
KHÔNG nằm ở kiến trúc.

### Ghi chú thí nghiệm TMA/BBA (mới nhất)
- Ý tưởng (paper "Bridge Then Begin Anew", AAAI-25): giữ v3 làm **bridge** (sinh
  nhãn giả + cho mượn trọng số đóng băng), **re-init** {listener_pool, fusion,
  fc_out}, train **thuần** trên candidate pool (nhãn giả), cắt bias mặt-đang-nói.
- Bridge khoẻ (in-domain eval-WAF 0.784). TMA Tier-1 → **61.47**, tụt 4.3.
- **Nguyên nhân:** confirmation-bias trap — distill từ nhãn giả cứng (mất soft
  info) = làm nghèo thầy; đồng thời vứt 9,395 mẫu train thật (vẫn hữu ích).
- **Kết luận:** dừng nhánh; KHÔNG làm Tier-2 (clustering refine) vì xây trên đúng
  nền vừa sập. Bài học: nền train thật quan trọng hơn tưởng; "begin anew" không hợp.

---

## 4. Blocks đang gặp

1. **Trần thông tin ~66.** Approach feature-fusion đã cạn; kiến trúc/feature/training
   đều không phá được.
2. **Audio/text non-transfer là bản chất**, không sửa bằng kiến trúc — chỉ khai
   thác được contagion yếu.
3. **Confirmation-bias trap** trong self-training: nhãn giả = niềm tin của chính
   model → distill lại làm nghèo đi (TMA vừa chứng minh).
4. **Không có nhãn target**; in-domain eval vô dụng → mọi quyết định phải qua
   Codabench (tốn lượt nộp, nhiễu ±3–4).
5. **Nền train thật quan trọng hơn tưởng** — bỏ nó đi (TMA) là hại.

---

## 5. Hướng nghiên cứu đề xuất (xếp theo kỳ vọng/chi phí)

| # | Hướng | Ceiling | Chi phí | Ghi chú |
|---|---|---|---|---|
| **1** | **Ensemble** multi-seed × multi-arch (v3/v6/v11) — *đang chạy* | +1–2 (→67–68 nhờ draw may) | Thấp | Giảm nhiễu + đa dạng lỗi. Đòn bẩy chắc chắn nhất còn lại. **Ưu tiên hoàn tất.** |
| **2** | **MLLM fine-tune** (LoRA, DialogueLLM-style, video frames + prompt vai) | **Cao — con đường thực tế nhất tới 70** | Cao (dự án nặng) | Đem world-knowledge/reasoning mà feature-fusion thiếu. Zero-shot 0.42 → fine-tune có thể nhảy vọt. |
| **3** | **Backbone video chuyên FER** thay CLIP (AffectNet/FER+ pretrained) | Trung bình (chưa thử) | Trung bình | Đánh trúng modality transfer duy nhất. CLIP general-purpose; feature FER-specialized có thể tách cảm xúc listener tốt hơn. **Chưa test — đáng thử.** |
| **4** | **Ước lượng test-prior** (BBSE) thay τ heuristic | +0.5–1 | Thấp | Thay τ=2.5 phỏng đoán bằng prior ước lượng từ candidate pool. Squeeze rẻ. |
| **5** | **Video-only strong + audio/text auxiliary nhẹ** | Trung bình | Trung bình | Re-frame theo "chỉ video transfer": tối đa hoá nhánh video, hạ vai audio/text xuống bổ trợ. |

**Đánh giá thẳng:** +4.2 để lên 70 là **vượt tầm ensemble đơn thuần**. Con đường
thực tế tới 70 gần như chắc chắn là **#2 (MLLM fine-tune)** — thứ duy nhất mang
thông tin ngoài không gian feature hiện tại. Nếu không đầu tư #2, kỳ vọng thực tế
của approach hiện tại là **66–67** (ensemble + #3/#4 squeeze).

---

## Phụ lục: tài sản code

| File | Vai trò |
|---|---|
| `toolkit/models/memocmt_v3.py` | Kiến trúc winner (65.78). **KHÔNG sửa v1.py.** |
| `toolkit/models/memocmt_v6/v11.py` | Aux-loss variants (train-only, eval==v3) |
| `submission.py` | `adjust_submission` (τ), `ensemble_submission`, `make_pseudo_corpus` |
| `main-release.py` | Train loop 5-fold CV + test1 inference |
| `run_ensemble.sh` | 13-model ensemble runner (v3×7, v6×3, v11×3) |
| `tma_finetune.py` + `run_tma.sh` | TMA/BBA (Tier-1) — **nhánh đã dừng (61.47)** |
| `mllm_predict.py` | MLLM API pipeline (zero-shot 0.42) |
