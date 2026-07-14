import math
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder


class MLPSeqEncoder(nn.Module):
    """Per-token projection (Linear + LN + GELU + Dropout). Keeps the clean
    contextualized structure of the WavLM/RoBERTa backbone instead of imposing
    a left-to-right recurrence learned on only ~9.3k samples."""
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)             # (B, T, output_dim), time structure kept


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class CLSTransformerEncoder(nn.Module):
    """Same video encoder as v13."""
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.proj  = nn.Linear(in_dim, hidden_dim)
        self.cls   = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.pe    = SinusoidalPE(hidden_dim)
        enc_layer  = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pe(x)
        x = self.encoder(x)
        return self.drop_out(x[:, 0, :])


class MemoCMTV25(nn.Module):
    """
    v13 with the speaker LSTM replaced by a per-token MLP projection.

    Only change vs v13: LSTMSeqEncoder -> MLPSeqEncoder for audio & text.
    WavLM/RoBERTa features are already contextualized by huge pretrained
    models; a 1-layer unidirectional LSTM learned on ~9.3k samples imposes a
    poor left-to-right filter that can corrupt that structure. A parameter-
    lean per-token projection keeps the backbone's clean temporal context.
    Video branch, speaker cross-attention, and fusion are identical to v13.
    """

    def __init__(self, args):
        super().__init__()
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout     = args.dropout
        hidden_dim  = args.hidden_dim
        self.grad_clip      = args.grad_clip
        self.feat_type      = getattr(args, 'feat_type',      'utt')
        self.speaker_drop_p = getattr(args, 'speaker_drop_p', 0.0)
        tf_layers           = getattr(args, 'tf_layers',      2)
        tf_heads            = getattr(args, 'tf_heads',       max(4, hidden_dim // 32))

        num_heads = max(1, hidden_dim // 64)

        # Speaker branches: per-token MLP projection (was LSTM in v13)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = MLPSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPSeqEncoder(text_dim,  hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)

        # Listener face branch (same as v13)
        self.video_encoder = CLSTransformerEncoder(
            in_dim=video_dim, hidden_dim=hidden_dim,
            num_heads=tf_heads, num_layers=tf_layers, dropout=dropout,
        )

        # Speaker bidir cross-attention (same as v13)
        self.cross_attn_a2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_a2t       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t       = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_t2a       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a       = nn.LayerNorm(hidden_dim)
        self.speaker_drop   = nn.Dropout(dropout)

        # Fusion (same as v13 -- unchanged; Block 2 intentionally NOT applied)
        self.cross_attn_fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fusion       = nn.LayerNorm(hidden_dim)
        self.fuse_drop         = nn.Dropout(dropout)

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        # Speaker encoding (per-token MLP; keeps time dimension)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)    # (B, Ta, H)
            h_t = self.text_encoder(text)      # (B, Tt, H)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)
            h_t = self.text_encoder(text).unsqueeze(1)

        # Listener face: CLS token (same as v13)
        listener_feat = self.video_encoder(video)

        # Speaker bidir cross-attention (same as v13)
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t     = self.norm_a2t(self.proj_a2t(a2t))
        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a     = self.norm_t2a(self.proj_t2a(t2a))

        sp_a = self.speaker_drop(a2t.mean(dim=1))
        sp_t = self.speaker_drop(t2a.mean(dim=1))

        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            sp_a = torch.zeros_like(sp_a)
            sp_t = torch.zeros_like(sp_t)

        # Fusion (same as v13)
        q  = listener_feat.unsqueeze(1)
        kv = torch.stack([sp_a, sp_t], dim=1)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(self.norm_fusion(attn_out.squeeze(1) + listener_feat))

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
