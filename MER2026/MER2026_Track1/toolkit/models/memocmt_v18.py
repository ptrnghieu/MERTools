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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiCLSTransformerEncoder(nn.Module):
    """
    Linear proj → prepend N learnable CLS tokens → (optional PE) → TransformerEncoder.
    Returns the N CLS outputs: (B, N, H).

    N CLS tokens give the video branch more capacity to hold multi-faceted
    listener affect (e.g. surprise + worry simultaneously) than a single CLS
    at hidden_dim=128.  use_pe=False makes it a permutation-invariant
    Bag-of-Frames aggregator (Cải tiến 1) — CLS tokens accumulate expression
    frequency/intensity rather than temporal order.
    """
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, dropout,
                 n_cls=2, use_pe=True):
        super().__init__()
        self.n_cls  = n_cls
        self.use_pe = use_pe
        self.proj   = nn.Linear(in_dim, hidden_dim)
        self.cls    = nn.Parameter(torch.zeros(1, n_cls, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.pe     = SinusoidalPE(hidden_dim) if use_pe else None
        enc_layer   = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x   = self.proj(x)                            # (B, T, H)
        cls = self.cls.expand(x.size(0), -1, -1)      # (B, N, H)
        x   = torch.cat([cls, x], dim=1)              # (B, N+T, H)
        if self.use_pe:
            x = self.pe(x)
        x   = self.encoder(x)                         # (B, N+T, H)
        return self.drop(x[:, :self.n_cls, :])        # (B, N, H) — N CLS tokens


class MemoCMTV18(nn.Module):
    """
    v13 + Multi-CLS tokens for the listener (video) branch.

    Changes vs v13:
      - Video branch: 1 CLS token → N learnable CLS tokens (default 2).
        Encoder returns (B, N, H) instead of (B, H).
      - Fusion: the N listener tokens (Query) attend the 2 speaker tokens;
        residual + LayerNorm keeps (B, N, H), then Flatten → (B, N*H).
      - Classifier: Linear(N*H → output_dim) sees all N tokens.
      - use_pe toggles positional encoding (True = v13 base; False = combine
        with Bag-of-Frames / Cải tiến 1).

    New yaml params (with defaults):
      n_cls:     2      number of learnable CLS tokens
      use_pe:    true   add sinusoidal PE to the video sequence
      tf_layers: 2
      tf_heads:  4
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
        self.n_cls          = getattr(args, 'n_cls',          2)
        use_pe              = bool(getattr(args, 'use_pe',    True))
        tf_layers           = getattr(args, 'tf_layers',      2)
        tf_heads            = getattr(args, 'tf_heads',       max(4, hidden_dim // 32))

        num_heads = max(1, hidden_dim // 64)  # speaker cross-attn / fusion

        # Speaker branches (same as v13)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)

        # Listener branch: multi-CLS transformer
        self.video_encoder = MultiCLSTransformerEncoder(
            in_dim=video_dim, hidden_dim=hidden_dim,
            num_heads=tf_heads, num_layers=tf_layers, dropout=dropout,
            n_cls=self.n_cls, use_pe=use_pe,
        )

        # Speaker bidir cross-attention (same as v13)
        self.cross_attn_a2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_a2t       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t       = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_t2a       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a       = nn.LayerNorm(hidden_dim)
        self.speaker_drop   = nn.Dropout(dropout)

        # Fusion: N listener tokens (Q) attend 2 speaker tokens (same mechanism as v13)
        self.cross_attn_fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fusion       = nn.LayerNorm(hidden_dim)
        self.fuse_drop         = nn.Dropout(dropout)

        # Flatten N tokens → classify
        flat_dim = self.n_cls * hidden_dim
        self.fc_out_1 = nn.Linear(flat_dim, output_dim1)
        self.fc_out_2 = nn.Linear(flat_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        # Speaker encoding (sequence → sequence)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)    # (B, Ta, H)
            h_t = self.text_encoder(text)      # (B, Tt, H)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)
            h_t = self.text_encoder(text).unsqueeze(1)

        # Listener encoding: N CLS tokens from Transformer
        listener_tokens = self.video_encoder(video)  # (B, N, H)

        # Speaker bidir cross-attention (same as v13)
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t     = self.norm_a2t(self.proj_a2t(a2t))
        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a     = self.norm_t2a(self.proj_t2a(t2a))

        sp_a = self.speaker_drop(a2t.mean(dim=1))   # (B, H)
        sp_t = self.speaker_drop(t2a.mean(dim=1))   # (B, H)

        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            sp_a = torch.zeros_like(sp_a)
            sp_t = torch.zeros_like(sp_t)

        # Fusion: N listener tokens attend 2 speaker tokens
        kv = torch.stack([sp_a, sp_t], dim=1)                    # (B, 2, H)
        attn_out, _ = self.cross_attn_fusion(query=listener_tokens, key=kv, value=kv)
        # residual with the listener tokens + LayerNorm, keep (B, N, H)
        fused = self.fuse_drop(self.norm_fusion(attn_out + listener_tokens))

        # Flatten N tokens → (B, N*H)
        features = fused.flatten(1)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
