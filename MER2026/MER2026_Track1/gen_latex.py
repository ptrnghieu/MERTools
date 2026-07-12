# -*- coding: utf-8 -*-
"""Generate a LaTeX report of all MER-Cross architectures (idea / rationale /
reference + confusion matrix + per-class metrics). Compile with XeLaTeX."""
import json

EMOS=['neutral','angry','happy','sad','worried','surprise']
EAB=['neu','ang','hap','sad','wor','sur']

CM=json.loads(r'''[{"name":"v6","waf":0.785,"cm":[[1635,178,156,96,161,58],[174,1681,36,71,109,84],[97,26,1662,14,11,17],[89,58,17,1390,81,23],[111,86,24,55,756,49],[23,52,21,12,37,245]],"per":{"neutral":{"p":0.768,"r":0.716,"f1":0.741,"n":2284},"angry":{"p":0.808,"r":0.78,"f1":0.794,"n":2155},"happy":{"p":0.867,"r":0.91,"f1":0.888,"n":1827},"sad":{"p":0.849,"r":0.838,"f1":0.843,"n":1658},"worried":{"p":0.655,"r":0.699,"f1":0.676,"n":1081},"surprise":{"p":0.515,"r":0.628,"f1":0.566,"n":390}}},
{"name":"v8","waf":0.7816,"cm":[[1625,164,148,108,166,73],[150,1683,34,70,114,104],[98,33,1648,12,16,20],[90,60,18,1389,74,27],[107,95,18,67,725,69],[31,46,15,10,30,258]],"per":{"neutral":{"p":0.773,"r":0.711,"f1":0.741,"n":2284},"angry":{"p":0.809,"r":0.781,"f1":0.795,"n":2155},"happy":{"p":0.876,"r":0.902,"f1":0.889,"n":1827},"sad":{"p":0.839,"r":0.838,"f1":0.838,"n":1658},"worried":{"p":0.644,"r":0.671,"f1":0.657,"n":1081},"surprise":{"p":0.468,"r":0.662,"f1":0.548,"n":390}}},
{"name":"v3","waf":0.7816,"cm":[[1588,196,150,100,184,66],[155,1708,38,71,108,75],[88,31,1657,13,21,17],[83,64,15,1385,86,25],[98,98,22,63,758,42],[24,54,22,12,36,242]],"per":{"neutral":{"p":0.78,"r":0.695,"f1":0.735,"n":2284},"angry":{"p":0.794,"r":0.793,"f1":0.793,"n":2155},"happy":{"p":0.87,"r":0.907,"f1":0.888,"n":1827},"sad":{"p":0.842,"r":0.835,"f1":0.839,"n":1658},"worried":{"p":0.635,"r":0.701,"f1":0.667,"n":1081},"surprise":{"p":0.518,"r":0.621,"f1":0.565,"n":390}}},
{"name":"v10 (CLIP+AU)","waf":0.7803,"cm":[[1526,224,134,119,192,89],[127,1720,26,61,116,105],[95,39,1630,17,23,23],[61,73,15,1412,79,18],[85,100,11,66,762,57],[26,51,12,11,23,267]],"per":{"neutral":{"p":0.795,"r":0.668,"f1":0.726,"n":2284},"angry":{"p":0.779,"r":0.798,"f1":0.789,"n":2155},"happy":{"p":0.892,"r":0.892,"f1":0.892,"n":1827},"sad":{"p":0.837,"r":0.852,"f1":0.844,"n":1658},"worried":{"p":0.638,"r":0.705,"f1":0.67,"n":1081},"surprise":{"p":0.478,"r":0.685,"f1":0.563,"n":390}}},
{"name":"v11","waf":0.7792,"cm":[[1593,218,142,103,168,60],[154,1694,39,72,112,84],[98,25,1663,11,16,14],[86,75,16,1376,85,20],[89,108,17,61,752,54],[27,62,20,15,28,238]],"per":{"neutral":{"p":0.778,"r":0.697,"f1":0.736,"n":2284},"angry":{"p":0.776,"r":0.786,"f1":0.781,"n":2155},"happy":{"p":0.877,"r":0.91,"f1":0.893,"n":1827},"sad":{"p":0.84,"r":0.83,"f1":0.835,"n":1658},"worried":{"p":0.648,"r":0.696,"f1":0.671,"n":1081},"surprise":{"p":0.506,"r":0.61,"f1":0.553,"n":390}}},
{"name":"v9 (CLIP+AU)","waf":0.7778,"cm":[[1583,200,149,107,167,78],[161,1691,30,81,103,89],[96,30,1643,16,20,22],[72,72,18,1400,74,22],[104,94,17,82,722,62],[26,51,15,11,27,260]],"per":{"neutral":{"p":0.775,"r":0.693,"f1":0.732,"n":2284},"angry":{"p":0.791,"r":0.785,"f1":0.788,"n":2155},"happy":{"p":0.878,"r":0.899,"f1":0.888,"n":1827},"sad":{"p":0.825,"r":0.844,"f1":0.835,"n":1658},"worried":{"p":0.649,"r":0.668,"f1":0.658,"n":1081},"surprise":{"p":0.488,"r":0.667,"f1":0.563,"n":390}}},
{"name":"v3 (FER)","waf":0.7713,"cm":[[1529,201,155,125,192,82],[125,1707,34,100,100,89],[97,40,1616,22,25,27],[75,67,19,1385,87,25],[100,93,16,80,744,48],[27,48,20,10,28,257]],"per":{"neutral":{"p":0.783,"r":0.669,"f1":0.722,"n":2284},"angry":{"p":0.792,"r":0.792,"f1":0.792,"n":2155},"happy":{"p":0.869,"r":0.885,"f1":0.877,"n":1827},"sad":{"p":0.804,"r":0.835,"f1":0.82,"n":1658},"worried":{"p":0.633,"r":0.688,"f1":0.659,"n":1081},"surprise":{"p":0.487,"r":0.659,"f1":0.56,"n":390}}},
{"name":"v7","waf":0.7184,"cm":[[1466,280,158,113,180,87],[233,1489,46,88,176,123],[104,35,1641,10,18,19],[120,96,21,1268,120,33],[111,132,25,74,677,62],[37,68,26,15,54,190]],"per":{"neutral":{"p":0.708,"r":0.642,"f1":0.673,"n":2284},"angry":{"p":0.709,"r":0.691,"f1":0.7,"n":2155},"happy":{"p":0.856,"r":0.898,"f1":0.877,"n":1827},"sad":{"p":0.809,"r":0.765,"f1":0.786,"n":1658},"worried":{"p":0.553,"r":0.626,"f1":0.587,"n":1081},"surprise":{"p":0.37,"r":0.487,"f1":0.42,"n":390}}}]''')
CMBY={d['name']:d for d in CM}

def cm_table(d):
    rows=d['cm']
    s=[r"\begin{table}[H]\centering\small",
       r"\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{l" + "c"*6 + "}",
       r"\toprule",
       r"\textbf{true\,$\downarrow$ / pred\,$\rightarrow$} & " + " & ".join(r"\textbf{%s}"%a for a in EAB) + r" \\",
       r"\midrule"]
    for i,row in enumerate(rows):
        tot=sum(row) or 1
        cells=[]
        for j,c in enumerate(row):
            pct=round(c/tot*100)
            shade=min(65,round(c/tot*70))
            val=r"\textbf{%d}"%pct if i==j else "%d"%pct
            cells.append(r"\cellcolor{cmblue!%d}%s"%(shade,val))
        s.append(r"\textbf{%s} & "%EAB[i] + " & ".join(cells) + r" \\")
    s+=[r"\bottomrule",r"\end{tabular}",
        r"\caption{Confusion matrix (row-normalised \%, in-domain held-out). Diagonal = recall.}",
        r"\end{table}"]
    return "\n".join(s)

def per_table(d):
    s=[r"\begin{table}[H]\centering\small",
       r"\begin{tabular}{lcccc}",r"\toprule",
       r"\textbf{Class} & \textbf{P} & \textbf{R} & \textbf{F1} & \textbf{N} \\",r"\midrule"]
    for e in EMOS:
        p=d['per'][e]
        s.append(r"%s & %.3f & %.3f & %.3f & %d \\"%(e,p['p'],p['r'],p['f1'],p['n']))
    s+=[r"\midrule",
        r"\textbf{Weighted-F1 (in-domain)} & \multicolumn{4}{c}{\textbf{%.4f}} \\"%d['waf'],
        r"\bottomrule",r"\end{tabular}",
        r"\caption{Per-class precision / recall / F1 (in-domain held-out).}",
        r"\end{table}"]
    return "\n".join(s)

# (id, name, idea, why, refs, waf_test, cm_key)
A=[
("v1","MemoCMT baseline",
 "Nhánh speaker: cross-attention hai chiều audio$\\leftrightarrow$text rồi nối và mean-pool thành một vector; nhánh listener: video qua Learnable Query Pooling; fusion: query = listener, key/value = speaker + residual.",
 "Tái lập MemoCMT làm baseline cho hội thoại dyadic: cross-attention để hai modality bổ sung nhau, listener video làm truy vấn vì đó là đối tượng cần dự đoán.",
 r"\cite{memocmt,vaswani2017}","baseline",None),
("v3","MemoCMT-v3 (best)",
 "v1 + \\textbf{Fix2} (mean-pool a2t và t2a \\emph{riêng} $\\to$ sp\\_a, sp\\_t) + \\textbf{Fix3} (fusion K/V = 2 token [sp\\_a, sp\\_t]).",
 "Fix2 sửa mất cân bằng khi số token audio$\\neq$text bị nuốt trong mean chung; Fix3 tránh attention suy biến khi chỉ có một token K/V. Đây là mô hình đạt WAF cao nhất.",
 r"\cite{memocmt,vaswani2017}","\\textbf{65.78}",'v3'),
("v2","Visual-anchored gated fusion",
 "Video (BiLSTM) self-attention rồi attend sang audio \\& text (query = visual); gate $g\\cdot v_a+(1-g)\\cdot v_t + h_v$ (residual video thô) rồi pooling.",
 "Ở test chỉ video (listener) là tín hiệu đáng tin và transfer được $\\to$ đặt video làm trục, audio/text chỉ bổ trợ, residual giữ video khi a/t không đáng tin.",
 r"\cite{arevalo2017gmu}","63--64",None),
("v4/v5","Full-sequence fusion (fixed / gated)",
 "Listener query attend \\emph{toàn chuỗi} a2t và t2a riêng (frame-level); v4 trộn cố định $0.5/0.5$, v5 dùng gate học per-channel.",
 "Thử tăng độ mịn frame-level trong fusion và đảm bảo cân bằng audio/text qua softmax riêng từng modality.",
 r"\cite{vaswani2017}","$\\sim$64",None),
("v6","CM-StEW auxiliary transfer",
 "Backbone = v3. Train thêm: MLP dịch listener\\_feat $\\to$ sp\\_a/sp\\_t (detach, L1) + căn chỉnh cosine listener$\\leftrightarrow$speaker. Chỉ lúc train.",
 "Trong train (Individual) cả ba modality cùng một người, nên audio/text có thể \\emph{dạy} video encoder mang thông tin cảm xúc; ở test nhánh video giàu hơn sẽ transfer.",
 r"\cite{hinton2015}","65.36",'v6'),
("v7","Nonverbal Conflict Exposure",
 "Backbone = v3. Lúc train, xác suất $p$ tráo audio+text của mẫu bằng của donor \\emph{khác nhãn} trong batch (giữ video + nhãn).",
 "Train có audio/text luôn nhất quán với nhãn; test thì thuộc speaker (có thể mâu thuẫn). Phơi bày mâu thuẫn để ép model neo vào video, không bị speaker sai vai đánh lừa.",
 r"\cite{ganin2016dann}","62--63",'v7'),
("v8","Reliability-Guided Complementation",
 "Video anchor truy hồi tham chiếu speaker (query = video); head phụ video-only cho độ tin cậy $c_v$; $h_v^*=h_v+(1-c_v)\\Delta$, head trên $[h_v^*,t_{ref},a_{ref}]$.",
 "Chỉ để speaker bổ trợ \\emph{khi} video không chắc chắn; video tự tin thì nén speaker $\\to$ giảm nhiễu từ modality không transfer.",
 r"\cite{arevalo2017gmu}","63",'v8'),
("v11","Deep-supervised video branch",
 "Backbone = v3. Train thêm CE phụ chỉ trên listener\\_feat (video-only head).",
 "In-domain audio/text mạnh nên fusion đạt loss thấp mà không cần video tốt (video ``lười''). Deep supervision ép nhánh video tự phân biệt cảm xúc $\\to$ anchor mạnh hơn cho test.",
 r"\cite{lee2015dsn}","$\\sim$65",'v11'),
("v9/v10","FACS Action-Unit features",
 "Feature video = CLIP $\\oplus$ AU (py-feat). v9: hai nhánh (CLIP-enc, AU-enc) $\\to$ gated embedding; v10: giữ CLIP làm anchor, au\\_feat thêm làm token K/V thứ 3.",
 "AU (FACS) là tín hiệu biểu cảm bất biến danh tính, kỳ vọng bổ sung cho CLIP. Thực tế bị \\emph{observation-shift}: speaker nói kích hoạt mouth-AU cấu âm, listener im lặng thì không $\\to$ hại.",
 r"\cite{ekman1978facs,cheong2023pyfeat}","60--61",'v9'),
("v12","Asymmetric VIB-FiLM fusion",
 "Video = anchor; speaker (sp\\_a,sp\\_t) nén qua Variational Information Bottleneck $\\to$ code $z$ $\\to$ sinh FiLM $(\\gamma,\\beta)$ điều chế listener\\_feat. FiLM zero-init; VIB KL = loss train-only.",
 "Bất đối xứng bản thể: video là bằng chứng trực tiếp, audio/text chỉ là kích thích ngoại lai. VIB lọc bỏ đặc trưng speaker không transfer, FiLM để speaker \\emph{điều chỉnh} chứ không lấn át video.",
 r"\cite{alemi2017,perez2018}","(đang chạy)",None),
("FER","FER-specialised video backbone",
 "Thay CLIP bằng embedding từ HSEmotion (EfficientNet-B2, pretrained AffectNet-8); kiến trúc v3 giữ nguyên audio/text/fusion.",
 "Kỳ vọng vision chuyên biểu cảm bắt vi biểu cảm listener tốt hơn CLIP generic. Thực tế thua CLIP: HSEmotion train trên mặt posed/biểu cảm mạnh $\\neq$ mặt listener hội thoại tinh tế (domain gap).",
 r"\cite{savchenko2022,mollahosseini2017}","59.6--61",'v3 (FER)'),
("MLLM","Qwen2.5-VL QLoRA + verbalizer",
 "Frame mặt listener + transcript speaker $\\to$ prompt vai $\\to$ LoRA sinh nhãn; infer bằng log-prob first-token của 6 nhãn (verbalizer). Biến thể video-only bỏ transcript.",
 "Thử đưa reasoning/world-knowledge của MLLM. Thất bại: vision generic (ViT freeze) trên crop $112^2$ đọc micro-expression yếu (video-only 42.8); in-domain 0.87 nhưng test 0.50 $\\to$ nút thắt là tri giác thị giác, không phải reasoning.",
 r"\cite{qwen25vl,dettmers2023,schick2021}","42--50 (in-dom 0.59--0.87)",None),
("Transductive","Pseudo-label / TMA-BBA",
 "Dùng 20k candidate không nhãn: (a) pseudo-label trộn vào train, retrain; (b) TMA/BBA: v3 làm bridge, re-init head, train thuần trên pseudo pool.",
 "Khai thác dữ liệu target không nhãn. Pseudo-mix chỉ +0.2 (talking-face vẫn lấn át); TMA/BBA $-4.3$ do bẫy thiên kiến xác nhận (distill nhãn giả cứng làm nghèo model).",
 r"\cite{liang2020shot,zhu2025bba}","66.0 / 61.47",None),
("Label-shift","Post-hoc calibration ($\\tau$ / MLLS)",
 "Hiệu chỉnh dịch nhãn hậu kỳ: $\\text{logit}-\\tau\\log(\\text{train\\_prior})$ ($\\tau=2.5$); hoặc BCTS + MLLS (EM ước lượng prior target trên 20k).",
 "Phân phối lớp train $\\neq$ test $\\to$ hiệu chỉnh prior. $\\tau$ giúp ổn định (+); MLLS có nguyên lý hơn nhưng bị vi phạm giả định label-shift (còn role-shift).",
 r"\cite{lipton2018,alexandari2020}","(hậu kỳ)",None),
]

BIB=r"""
\begin{thebibliography}{99}
\bibitem{memocmt} \emph{MemoCMT: Multimodal emotion recognition using cross-modal transformer-based feature fusion}, 2025.
\bibitem{vaswani2017} A. Vaswani et al., \emph{Attention Is All You Need}, NeurIPS 2017.
\bibitem{arevalo2017gmu} J. Arevalo et al., \emph{Gated Multimodal Units for Information Fusion}, ICLR Workshop 2017.
\bibitem{hinton2015} G. Hinton, O. Vinyals, J. Dean, \emph{Distilling the Knowledge in a Neural Network}, 2015.
\bibitem{ganin2016dann} Y. Ganin et al., \emph{Domain-Adversarial Training of Neural Networks}, JMLR 2016.
\bibitem{lee2015dsn} C.-Y. Lee et al., \emph{Deeply-Supervised Nets}, AISTATS 2015.
\bibitem{ekman1978facs} P. Ekman, W. Friesen, \emph{Facial Action Coding System}, 1978.
\bibitem{cheong2023pyfeat} J. Cheong et al., \emph{Py-Feat: Python Facial Expression Analysis Toolbox}, 2023.
\bibitem{alemi2017} A. Alemi et al., \emph{Deep Variational Information Bottleneck}, ICLR 2017.
\bibitem{perez2018} E. Perez et al., \emph{FiLM: Visual Reasoning with a General Conditioning Layer}, AAAI 2018.
\bibitem{savchenko2022} A. Savchenko, \emph{HSEmotion: Facial expression recognition with EfficientNet on AffectNet}, 2022.
\bibitem{mollahosseini2017} A. Mollahosseini et al., \emph{AffectNet: A Database for Facial Expression, Valence, and Arousal}, IEEE TAC 2017.
\bibitem{qwen25vl} Qwen Team, \emph{Qwen2.5-VL Technical Report}, 2025.
\bibitem{dettmers2023} T. Dettmers et al., \emph{QLoRA: Efficient Finetuning of Quantized LLMs}, NeurIPS 2023.
\bibitem{schick2021} T. Schick, H. Sch\"utze, \emph{Exploiting Cloze-Questions for Few-Shot Text Classification (PET)}, EACL 2021.
\bibitem{liang2020shot} J. Liang et al., \emph{Do We Really Need to Access the Source Data? (SHOT)}, ICML 2020.
\bibitem{zhu2025bba} J. Zhu et al., \emph{Bridge Then Begin Anew: Generating Target-Relevant Intermediate Model for Source-Free VER}, AAAI 2025.
\bibitem{lipton2018} Z. Lipton et al., \emph{Detecting and Correcting for Label Shift (BBSE)}, ICML 2018.
\bibitem{alexandari2020} A. Alexandari et al., \emph{Maximum Likelihood with Bias-Corrected Calibration (MLLS)}, ICML 2020.
\bibitem{mer2026} \emph{MER 2026: From Discriminative Emotion Recognition to Generative Emotion Understanding}, arXiv:2604.19417.
\end{thebibliography}
"""

HEAD=r"""% Compile with XeLaTeX hoặc LuaLaTeX (Unicode + tiếng Việt).
\documentclass[10pt]{article}
\usepackage{fontspec}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{booktabs,array,float,xcolor,colortbl}
\usepackage[hidelinks]{hyperref}
\definecolor{cmblue}{RGB}{63,69,168}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\title{\textbf{MER-Cross (MER2026 Track-1)\\ Tổng hợp kiến trúc: ý tưởng, cơ sở, và kết quả}}
\author{}\date{}
\begin{document}\maketitle
\noindent\textbf{Bài toán.} Dự đoán cảm xúc người \emph{nghe} (listener) trong hội thoại tay đôi.
Train = 9\,395 mẫu Individual (audio/text/video cùng người, đang nói); test = 574 mẫu
Interlocutor (video = listener im lặng, audio/text = speaker khác). Metric = weighted-F1 (WAF)
trên Codabench (nhiễu $\pm3$--4). Baseline tốt nhất \textbf{v3 = 65.78}~\cite{mer2026}.

\medskip\noindent\textbf{Lưu ý.} Confusion matrix \& metric per-class là \emph{in-domain}
(held-out train), do test không có nhãn (Codabench chỉ trả WAF tổng hợp). Chúng cho biết
model nhầm lớp nào trên train; WAF test đo khả năng transfer cross-role.
"""

SUMMARY=r"""
\section*{Bảng tổng hợp kết quả}
\begin{table}[H]\centering\small
\begin{tabular}{lll}
\toprule
\textbf{Kiến trúc} & \textbf{WAF test} & \textbf{Ghi chú} \\ \midrule
Pseudo-label (mix)        & 66.0 & transductive, +0.2 \\
v3 ensemble (7-seed)      & 65.80 & giảm variance \\
\textbf{v3 (best model)}  & \textbf{65.78} & Fix2+Fix3 \\
v6 CM-StEW                & 65.36 & aux transfer \\
v11 deep-sup video        & $\sim$65 & aux CE video \\
v4/v5 fusion variants     & $\sim$64 & \\
v2 visual-anchor          & 63--64 & \\
v8 reliability-gate       & 63 & \\
v7 conflict exposure      & 62--63 & \\
TMA/BBA transductive      & 61.47 & confirmation bias \\
FER backbone (HSEmotion)  & 59.6--61 & domain gap \\
v9/v10 AU feature         & 60--61 & observation-shift \\
MLLM SFT (+transcript)    & 49.84 & in-dom 0.87 \\
MLLM video-only           & 42.8 & in-dom 0.59 \\
MLLM zero-shot            & 42 & \\
\bottomrule
\end{tabular}
\caption{WAF Codabench (test, 574 mẫu). Trần thông tin $\sim$66 bền vững mọi trục.}
\end{table}
"""

def sec(a):
    aid,name,idea,why,refs,waf,cmk=a
    s=[r"\section*{%s \quad\normalsize\textnormal{[%s]} \hfill \small WAF test: %s}"%(name,aid,waf),
       r"\textbf{Ý tưởng.}\ %s"%idea,
       r"\textbf{Vì sao.}\ %s"%why,
       r"\textbf{Tham khảo.}\ %s"%refs]
    if cmk and cmk in CMBY:
        d=CMBY[cmk]
        s.append(cm_table(d)); s.append(per_table(d))
    else:
        s.append(r"\emph{(Không có confusion matrix in-domain: cv npz không còn / pipeline riêng / đang chạy.)}")
    return "\n\n".join(s)

doc=HEAD+SUMMARY+"\n"+"\n\n".join(sec(a) for a in A)+"\n\n"+BIB+"\n\\end{document}\n"
open('report_architectures.tex','w').write(doc)
print("wrote report_architectures.tex  (%d chars)"%len(doc))
