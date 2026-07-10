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

        }
        self.model = MODEL_MAP[args.model](args)

    def forward(self, batch):
        return self.model(batch)
