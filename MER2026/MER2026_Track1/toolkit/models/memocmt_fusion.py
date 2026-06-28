import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder


class LSTMSeqEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, dropout, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.dropout(out)


class LearnableQueryPooling(nn.Module):
    """Single learnable query that attends over a sequence → (B, H)."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x):
        # x: (B, T, H)
        q = self.query.expand(x.size(0), -1, -1)   # (B, 1, H)
        out, _ = self.attn(query=q, key=x, value=x) # (B, 1, H)
        return out.squeeze(1)                        # (B, H)


class MemoCMTFusion(nn.Module):
    """
    Speaker branch : bidirectional cross-attention on audio + text (MemoCMT-style).
    Listener branch: LSTM + learnable query pooling on video.
    Fusion          : concat(speaker, listener) → MLP → classifier.
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

        self.feat_type = getattr(args, 'feat_type', 'utt')
        num_heads = max(1, hidden_dim // 64)

        # ── Encoders ─────────────────────────────────────────────────────
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMSeqEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        # ── Speaker branch: bidirectional cross-attention ─────────────────
        # Block 1: Q=audio attends over text
        self.cross_attn_a2t = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.proj_a2t = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t = nn.LayerNorm(hidden_dim)

        # Block 2: Q=text attends over audio
        self.cross_attn_t2a = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.proj_t2a = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a = nn.LayerNorm(hidden_dim)

        self.speaker_drop = nn.Dropout(dropout)

        # ── Listener branch: learnable query pooling ──────────────────────
        self.listener_pool = LearnableQueryPooling(hidden_dim, num_heads, dropout)

        # ── Fusion + classification head ──────────────────────────────────
        fused_dim = hidden_dim * 2
        self.fc1      = nn.Linear(fused_dim, hidden_dim)
        self.act      = nn.ReLU()
        self.fuse_drop = nn.Dropout(dropout)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        # ── Encode ───────────────────────────────────────────────────────
        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)   # (B, T_a, H)
            h_t = self.text_encoder(text)     # (B, T_t, H)
            h_v = self.video_encoder(video)   # (B, T_v, H)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)  # (B, 1, H)
            h_t = self.text_encoder(text).unsqueeze(1)    # (B, 1, H)
            h_v = self.video_encoder(video).unsqueeze(1)  # (B, 1, H)

        # ── Speaker: bidirectional cross-attention ────────────────────────
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t     = self.norm_a2t(self.proj_a2t(a2t))       # (B, T_a, H)

        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a     = self.norm_t2a(self.proj_t2a(t2a))       # (B, T_t, H)

        speaker_seq  = torch.cat([a2t, t2a], dim=1)       # (B, T_a+T_t, H)
        speaker_feat = self.speaker_drop(speaker_seq.mean(dim=1))  # (B, H)

        # ── Listener: learnable query pooling ─────────────────────────────
        listener_feat = self.listener_pool(h_v)            # (B, H)

        # ── Fusion ────────────────────────────────────────────────────────
        fused    = torch.cat([speaker_feat, listener_feat], dim=-1)  # (B, 2H)
        features = self.fuse_drop(self.act(self.fc1(fused)))          # (B, H)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()

        return features, emos_out, vals_out, interloss
