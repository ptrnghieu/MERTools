import math
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder


class LSTMSeqEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, dropout, num_layers=1):
        super().__init__()
        self.input_drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=False)
        self.output_drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.rnn(self.input_drop(x))
        return self.output_drop(out)


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class VideoTransformerEncoder(nn.Module):
    """
    Linear proj → prepend CLS → sinusoidal PE → TransformerEncoder.
    Returns (cls_out, frame_seq): CLS token and full frame sequence separately.
    """
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.cls  = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.pe   = SinusoidalPE(hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x   = self.proj(x)                            # (B, T, H)
        cls = self.cls.expand(x.size(0), -1, -1)      # (B, 1, H)
        x   = torch.cat([cls, x], dim=1)              # (B, T+1, H)
        x   = self.pe(x)
        x   = self.encoder(x)                         # (B, T+1, H)
        return self.drop(x[:, 0, :]), self.drop(x[:, 1:, :])  # (B,H), (B,T,H)


class MemoCMTV14(nn.Module):
    """
    v13 + Temporal Cross-Attention fusion (keep full speaker sequences).

    Changes vs v13:
      - No early mean-pool of speaker (a2t, t2a): keep full sequences F_a, F_t.
      - F_context = concat([F_a, F_t], dim=1)   -> (B, Ta+Tt, H)
      - Fusion cross-attn: Q = video_frames (B, Tv, H),  K/V = F_context
        -> F_fused_seq  (B, Tv, H)
      - Global avg-pool over Tv, residual with CLS token, LayerNorm -> features

    Video branch and speaker LSTM/cross-attn are identical to v13.
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

        num_heads = max(1, hidden_dim // 64)  # speaker cross-attn / fusion heads

        # Speaker branches (same as v3/v13)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)

        # Listener branch (same transformer as v13, but returns frame seq too)
        self.video_encoder = VideoTransformerEncoder(
            in_dim=video_dim, hidden_dim=hidden_dim,
            num_heads=tf_heads, num_layers=tf_layers, dropout=dropout,
        )

        # Speaker bidir cross-attention (same as v3/v13)
        self.cross_attn_a2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_a2t       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t       = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_t2a       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a       = nn.LayerNorm(hidden_dim)
        self.speaker_drop   = nn.Dropout(dropout)

        # Temporal fusion: Q=video_frames, K/V=F_context (full speaker seqs)
        self.cross_attn_fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fusion       = nn.LayerNorm(hidden_dim)
        self.fuse_drop         = nn.Dropout(dropout)

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        # Speaker encoding
        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)    # (B, Ta, H)
            h_t = self.text_encoder(text)      # (B, Tt, H)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)
            h_t = self.text_encoder(text).unsqueeze(1)

        # Listener: CLS summary + full frame sequence
        listener_cls, frames_v = self.video_encoder(video)  # (B,H), (B,Tv,H)

        # Speaker bidir cross-attention — keep FULL sequences
        F_a, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        F_a     = self.norm_a2t(self.proj_a2t(F_a))    # (B, Ta, H)
        F_t, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        F_t     = self.norm_t2a(self.proj_t2a(F_t))    # (B, Tt, H)

        # Optional: zero out entire speaker context during training
        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            F_a = torch.zeros_like(F_a)
            F_t = torch.zeros_like(F_t)
        else:
            F_a = self.speaker_drop(F_a)
            F_t = self.speaker_drop(F_t)

        # Merge speaker context: (B, Ta+Tt, H)
        F_context = torch.cat([F_a, F_t], dim=1)

        # Temporal cross-attention: each video frame queries speaker context
        F_fused, _ = self.cross_attn_fusion(query=frames_v, key=F_context, value=F_context)
        # (B, Tv, H)

        # Global avg-pool + residual (CLS token) + LayerNorm
        pooled   = F_fused.mean(dim=1)                            # (B, H)
        features = self.fuse_drop(self.norm_fusion(pooled + listener_cls))  # (B, H)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
