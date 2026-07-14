"""
get_models: get models and load default configs; 
link: https://github.com/thuiar/MMSA-FET/tree/master
"""
import torch.nn as nn

from .tfn import TFN
from .lmf import LMF
from .mfn import MFN
from .mfm import MFM
from .mult import MULT
from .misa import MISA
from .mctn import MCTN
from .mmim import MMIM
from .lf_dnn import LF_DNN
from .ef_lstm import EF_LSTM
from .graph_mfn import Graph_MFN
from .attention import Attention
from .attention_topn import Attention_TOPN
from .cross_role_attention import CrossRoleAttention
from .two_stage_model import TwoStageModel
from .grasp_sequence_fusion import GRASPSequenceFusion
from .speaker_listener_fusion import SpeakerListenerFusion
from .memocmt_fusion import MemoCMTFusion
from .memocmt_v1 import MemoCMTV1
from .memocmt_mmin import MemoCMTMMIN
from .memocmt_mmin_hybrid import MemoCMTMMINHybrid
from .memocmt_v2 import MemoCMTV2
from .memocmt_v3 import MemoCMTV3
from .memocmt_v4 import MemoCMTV4
from .memocmt_v5 import MemoCMTV5
from .memocmt_v6 import MemoCMTV6
from .memocmt_v7 import MemoCMTV7
from .memocmt_v8 import MemoCMTV8
from .memocmt_v9 import MemoCMTV9
from .memocmt_v11 import MemoCMTV11
from .memocmt_v10 import MemoCMTV10
from .memocmt_v12 import MemoCMTV12
from .memocmt_v13 import MemoCMTV13
from .memocmt_v14 import MemoCMTV14
from .memocmt_v15 import MemoCMTV15
from .memocmt_v16 import MemoCMTV16
from .memocmt_v17 import MemoCMTV17
from .memocmt_v18 import MemoCMTV18
from .memocmt_v19 import MemoCMTV19
from .memocmt_v20 import MemoCMTV20
from .memocmt_v21 import MemoCMTV21
from .memocmt_v22 import MemoCMTV22
from .memocmt_v23 import MemoCMTV23
from .memocmt_v24 import MemoCMTV24
from .memocmt_v25 import MemoCMTV25
from .memocmt_v26 import MemoCMTV26
from .memocmt_v27 import MemoCMTV27

class get_models(nn.Module):
    def __init__(self, args):
        super(get_models, self).__init__()
        # misa/mmim在有些参数配置下会存在梯度爆炸的风险
        # tfn 显存占比比较高

        MODEL_MAP = {
            
            # 特征压缩到句子级再处理，所以支持 utt/align/unalign
            'attention': Attention,
            'lf_dnn': LF_DNN,
            'lmf': LMF,
            'misa': MISA,
            'mmim': MMIM,
            'tfn': TFN,
            
            # 只支持align
            'mfn': MFN, # slow
            'graph_mfn': Graph_MFN, # slow
            'ef_lstm': EF_LSTM, 
            'mfm': MFM, # slow
            'mctn': MCTN, # slow

            # 支持align/unalign
            'mult': MULT, # slow


            # 支持每个模态选择topn特征输入
            'attention_topn': Attention_TOPN,

            # cross-role transferability: shuffle audio/text during training
            'cross_role_attention': CrossRoleAttention,
            'two_stage_model': TwoStageModel,

            # video-guided cross-attention fusion for MER-Cross (frm_unalign)
            'grasp_sequence_fusion': GRASPSequenceFusion,

            # speaker (audio+text) → listener (video) cross-attention
            'speaker_listener_fusion': SpeakerListenerFusion,

            # MemoCMT-style bidir cross-attn speaker branch + learnable query pooling listener branch
            'memocmt_fusion': MemoCMTFusion,
            'memocmt_v1':     MemoCMTV1,

            # bare MMIN: imagine listener's own audio/text from video (Interlocutor domain shift)
            'memocmt_mmin':   MemoCMTMMIN,

            # hybrid MMIN: v1 speaker branch + imagined-listener branch fused together
            'memocmt_mmin_hybrid': MemoCMTMMINHybrid,

            # visual-anchored: bidir visual + self-attn + gated dual cross-attn (Q=visual)
            'memocmt_v2': MemoCMTV2,

            # v1 + per-modality speaker pooling + 2-token fusion
            'memocmt_v3': MemoCMTV3,

            # v3 fusion upgraded: listener attends full a2t/t2a sequences separately, 1:1 combine
            'memocmt_v4': MemoCMTV4,

            # v4 + learned per-channel gate for audio/text mix
            'memocmt_v5': MemoCMTV5,

            # v3 + CM-StEW auxiliary (train-only): translation + alignment
            # losses that distill speaker audio/text into the video encoder
            'memocmt_v6': MemoCMTV6,

            # v3 + Nonverbal Conflict Exposure (train-only): in-batch donor
            # swap of speaker audio/text to teach video-anchored robustness
            'memocmt_v7': MemoCMTV7,

            # VISAFF-style Reliability-Guided Affective Complementation:
            # video anchor + visual-guided speaker retrieval gated by video conf
            'memocmt_v8': MemoCMTV8,

            # separate CLIP + AU/landmark branches, gated embedding fusion
            'memocmt_v9': MemoCMTV9,

            # non-disruptive AU add-on: pure-CLIP anchor (v3) + au_feat as a
            # 3rd fusion token the attention may use or ignore
            'memocmt_v10': MemoCMTV10,

            # v3 + auxiliary video-only CE (deep-supervise the video anchor)
            'memocmt_v11': MemoCMTV11,

            # v3 + asymmetric VIB-FiLM fusion: video anchor modulated by a
            # VIB-compressed speaker (audio/text) context (not symmetric mixing)
            'memocmt_v12': MemoCMTV12,

            # v3 + Transformer Encoder + CLS token for listener (video) branch
            'memocmt_v13': MemoCMTV13,

            # v13 + Temporal Cross-Attention fusion: keep full speaker seqs,
            # Q=video_frames, K/V=concat(F_a,F_t), avg-pool + residual CLS
            'memocmt_v14': MemoCMTV14,

            # Multimodal Bottleneck Tokens: N learnable tokens mediate all
            # cross-modal interaction (no direct modality-to-modality attention)
            # Step1: bn queries speaker(a+t), Step2: bn queries listener video
            'memocmt_v15': MemoCMTV15,

            # MBT v2: flatten N tokens -> (B, N*H) + LayerNorm before classifier
            # hidden_dim=256, n_bottleneck=16 -> 4096-dim pre-classifier space
            'memocmt_v16': MemoCMTV16,

            # v13 + Bag-of-Frames: remove PE from video transformer so CLS
            # accumulates expression frequency/intensity, not temporal order
            'memocmt_v17': MemoCMTV17,

            # v13 + Multi-CLS: N learnable CLS tokens for the video branch
            # (default 2), fused then flattened -> N*H classifier input.
            # use_pe toggles PE (combine with Bag-of-Frames when False)
            'memocmt_v18': MemoCMTV18,

            # v13 + Learnable Positional Embedding + Multi-CLS tokens (keep PE):
            # fixed sinusoidal -> learnable PE table, N CLS tokens each learning
            # a different temporal aspect, flattened -> N*H classifier input
            'memocmt_v19': MemoCMTV19,

            # v13 + RoPE for the video branch: nn.TransformerEncoder + sinusoidal
            # -> custom RoPEAttentionLayer stack (relative position, no absolute
            # PE), robust to train/test video-length shift
            'memocmt_v20': MemoCMTV20,

            # v13 + attention-pooling for speaker (learnable query) instead of
            # mean-pool: focuses on salient speaker frames, output stays static
            'memocmt_v21': MemoCMTV21,

            # v13 + listener body-language branch (pose + optical flow), gated
            # into the listener representation (--body_feature required)
            'memocmt_v22': MemoCMTV22,

            # v22 redesigned: body as a K/V token (pure face Query), body encoder
            # = frame-wise MLP + max-pool instead of LSTM+mean (--body_feature)
            'memocmt_v23': MemoCMTV23,

            # v13 + FiLM fusion (GP1) + modality dropout (GP2): capacity-reducing
            # fusion (scale&shift vs 2-token MHA) + drop one speaker modality
            'memocmt_v24': MemoCMTV24,

            # v13 with speaker LSTM -> per-token MLP projection (keeps the
            # pretrained backbone's clean context; fewer params, less overfit)
            'memocmt_v25': MemoCMTV25,

            # v13 + Delta video features: subtract consecutive frames to
            # suppress static identity (attacks the identity/cross-role shortcut)
            'memocmt_v26': MemoCMTV26,

            # v13 + dual-pooling K/V: fusion K/V = [sp_a_mean, sp_a_max,
            # sp_t_mean, sp_t_max] (4 tokens), MHA mechanism unchanged, 0 new params
            'memocmt_v27': MemoCMTV27,

        }
        self.model = MODEL_MAP[args.model](args)

    def forward(self, batch):
        return self.model(batch)
