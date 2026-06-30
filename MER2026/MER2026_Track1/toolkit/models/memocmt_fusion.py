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


class MultiQueryAttentionPooling(nn.Module):
    """Multiple learnable queries attend over sequence, then mean-aggregated → (B, H)."""

    def __init__(self, hidden_dim, num_heads, dropout, num_queries=4):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.agg_drop = nn.Dropout(dropout)

    def forward(self, x):
        q = self.queries.expand(x.size(0), -1, -1)   # (B, Q, H)
        out, _ = self.attn(query=q, key=x, value=x)   # (B, Q, H)
        return self.agg_drop(out.mean(dim=1))          # (B, H)


class MemoCMTFusion(nn.Module):
    """
    Speaker branch : LSTM → bidir cross-attention (audio↔text) → mean pool
    Listener branch: LSTM → multi-query attention pooling
    Fusion         : Q=listener, K/V=speaker cross-attention + residual
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
        self.grad_clip = args.grad_clip

        self.feat_type      = getattr(args, 'feat_type',      'utt')
        self.speaker_drop_p = getattr(args, 'speaker_drop_p', 0.0)

        num_heads = max(1, hidden_dim // 64)

        # ── Encoders ──────────────────────────────────────────────────────
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMSeqEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        # ── Speaker: bidirectional cross-attention ─────────────────────────
        self.cross_attn_a2t = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.proj_a2t = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t = nn.LayerNorm(hidden_dim)

        self.cross_attn_t2a = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.proj_t2a = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a = nn.LayerNorm(hidden_dim)

        self.speaker_drop = nn.Dropout(dropout)

        # ── Listener: multi-query attention pooling ────────────────────────
        self.listener_pool = MultiQueryAttentionPooling(hidden_dim, num_heads, dropout, num_queries=4)

        # ── Fusion ────────────────────────────────────────────────────────
        self.cross_attn_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_fusion = nn.LayerNorm(hidden_dim)
        self.fuse_drop   = nn.Dropout(dropout)

        # ── Classifier ────────────────────────────────────────────────────
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        # ── Encode ────────────────────────────────────────────────────────
        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)   # (B, T_a, H)
            h_t = self.text_encoder(text)     # (B, T_t, H)
            h_v = self.video_encoder(video)   # (B, T_v, H)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)  # (B, 1, H)
            h_t = self.text_encoder(text).unsqueeze(1)    # (B, 1, H)
            h_v = self.video_encoder(video).unsqueeze(1)  # (B, 1, H)

        # ── Speaker: bidir cross-attention → mean pool ────────────────────
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t     = self.norm_a2t(self.proj_a2t(a2t))           # (B, T_a, H)

        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a     = self.norm_t2a(self.proj_t2a(t2a))           # (B, T_t, H)

        speaker_seq  = torch.cat([a2t, t2a], dim=1)           # (B, T_a+T_t, H)
        speaker_feat = self.speaker_drop(speaker_seq.mean(dim=1))  # (B, H)

        # ── Modality dropout (training only) ──────────────────────────────
        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            speaker_feat = torch.zeros_like(speaker_feat)

        # ── Listener: multi-query attention pooling ────────────────────────
        listener_feat = self.listener_pool(h_v)                # (B, H)

        # ── Fusion: Q=listener, K/V=speaker + residual ────────────────────
        q  = listener_feat.unsqueeze(1)   # (B, 1, H)
        kv = speaker_feat.unsqueeze(1)    # (B, 1, H)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(
            self.norm_fusion(attn_out.squeeze(1) + listener_feat)
        )                                                       # (B, H)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()

        return features, emos_out, vals_out, interloss
