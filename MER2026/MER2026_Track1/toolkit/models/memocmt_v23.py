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


class SimpleBodyEncoder(nn.Module):
    """Frame-wise MLP + max-pool (no RNN). Low capacity to resist overfitting on
    raw keypoint/flow coordinates; max-pool keeps peak movements (a sharp nod /
    shrug) that mean-pool would wash out."""
    def __init__(self, in_dim, hidden_dim, dropout=0.4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.mlp(x)                 # (B, Tb, H)
        x, _ = torch.max(x, dim=1)      # (B, H) — peak movement
        return x


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
    """Same video (face) encoder as v13."""
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


class MemoCMTV23(nn.Module):
    """
    v22 redesigned: body as a Key/Value token, not fused into the Query.

    Fixes vs v22:
      1. Query stays the pure face CLS (face_feat) -- never contaminated by the
         noisy body feature. Body enters only as a 3rd K/V token alongside the
         speaker tokens, so attention can down-weight it to ~0 when it's noisy
         (clean fallback to v13 behaviour). Residual is on the pure face_feat.
      2. Body encoder: LSTM+mean -> SimpleBodyEncoder (frame-wise MLP + max-pool)
         -- lower capacity (less overfit on raw coords) and keeps peak movements.

    Fusion:  Q = face_feat ;  K,V = [sp_a, sp_t, body_feat]  (B, 3, H)
             features = LN(attn_out + face_feat)

    Speaker branch and face branch identical to v13.
    """

    POSE_DIM = 27
    FLOW_DIM = 16

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
        body_dropout        = getattr(args, 'body_dropout',   0.4)
        tf_layers           = getattr(args, 'tf_layers',      2)
        tf_heads            = getattr(args, 'tf_heads',       max(4, hidden_dim // 32))

        num_heads = max(1, hidden_dim // 64)

        # Speaker branches (same as v13)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)

        # Listener face branch (same as v13)
        self.video_encoder = CLSTransformerEncoder(
            in_dim=video_dim, hidden_dim=hidden_dim,
            num_heads=tf_heads, num_layers=tf_layers, dropout=dropout,
        )

        # Listener body branch: simple frame-wise MLP + max-pool
        self.pose_encoder = SimpleBodyEncoder(self.POSE_DIM, hidden_dim, body_dropout)
        self.flow_encoder = SimpleBodyEncoder(self.FLOW_DIM, hidden_dim, body_dropout)
        self.body_norm    = nn.LayerNorm(hidden_dim)

        # Speaker bidir cross-attention (same as v13)
        self.cross_attn_a2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_a2t       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t       = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_t2a       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a       = nn.LayerNorm(hidden_dim)
        self.speaker_drop   = nn.Dropout(dropout)

        # Fusion: Q=face, K/V = [sp_a, sp_t, body]
        self.cross_attn_fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fusion       = nn.LayerNorm(hidden_dim)
        self.fuse_drop         = nn.Dropout(dropout)

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def _encode_body(self, body):
        pose = body[:, :, :self.POSE_DIM]
        flow = body[:, :, self.POSE_DIM:self.POSE_DIM + self.FLOW_DIM]
        pose_h = self.pose_encoder(pose)               # (B, H)
        flow_h = self.flow_encoder(flow)               # (B, H)
        return self.body_norm(pose_h + flow_h)         # (B, H)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        # Speaker encoding
        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)
            h_t = self.text_encoder(text)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)
            h_t = self.text_encoder(text).unsqueeze(1)

        # Listener face: pure CLS token (kept clean for the Query)
        face_feat = self.video_encoder(video)          # (B, H)

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

        # Build K/V token set: speaker (a, t) + optional listener body
        kv_tokens = [sp_a, sp_t]
        if 'bodys' in batch:
            kv_tokens.append(self._encode_body(batch['bodys']))
        kv = torch.stack(kv_tokens, dim=1)             # (B, 2 or 3, H)

        # Fusion: pure face Query attends the K/V set; residual on pure face
        q = face_feat.unsqueeze(1)                     # (B, 1, H)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(self.norm_fusion(attn_out.squeeze(1) + face_feat))

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
