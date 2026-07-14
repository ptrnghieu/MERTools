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


class DeltaCLSTransformerEncoder(nn.Module):
    """v13 video encoder + Delta (first temporal derivative) high-pass filter.

    delta[t] = V[t] - V[t-1] (delta[0] = 0) removes the static component
    (identity: face structure, skin tone) shared between consecutive frames,
    keeping facial dynamics (frown/smile/lip motion). This attacks the identity
    shortcut: in train speaker==listener so the model can recognise the ACTOR
    instead of the EXPRESSION; at test the actor is unseen and that shortcut
    breaks. A frown's delta pattern is similar across people -> generalizes.

    delta_mode='only'   -> project delta only (identity fully suppressed)
    delta_mode='concat' -> project [raw, delta] (keep static context + dynamics)
    """
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, dropout, delta_mode='only'):
        super().__init__()
        self.delta_mode = delta_mode
        proj_in = in_dim * 2 if delta_mode == 'concat' else in_dim
        self.proj  = nn.Linear(proj_in, hidden_dim)
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
        # x: (B, Tv, in_dim). Delta along time.
        delta = x[:, 1:, :] - x[:, :-1, :]
        zero  = torch.zeros(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        delta = torch.cat([zero, delta], dim=1)                # (B, Tv, in_dim)
        feat  = torch.cat([x, delta], dim=-1) if self.delta_mode == 'concat' else delta

        h   = self.proj(feat)                                  # (B, Tv, H)
        cls = self.cls.expand(h.size(0), -1, -1)
        h   = torch.cat([cls, h], dim=1)
        h   = self.pe(h)
        h   = self.encoder(h)
        return self.drop_out(h[:, 0, :])


class MemoCMTV26(nn.Module):
    """
    v13 + Delta video features (identity-bias high-pass filter).

    Only change vs v13: the video (face) encoder subtracts consecutive-frame
    features to suppress static identity before the CLS transformer. Speaker
    branch and fusion identical to v13.

    New yaml param: delta_mode ('only' | 'concat', default 'only').
    NB: run with a lower --video_feat_scale (e.g. 4) so there are enough frames
    for the temporal difference to carry signal (at scale 12 video is ~2 frames).
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
        delta_mode          = getattr(args, 'delta_mode',     'only')
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

        # Listener face branch: Delta + CLS transformer
        self.video_encoder = DeltaCLSTransformerEncoder(
            in_dim=video_dim, hidden_dim=hidden_dim,
            num_heads=tf_heads, num_layers=tf_layers, dropout=dropout,
            delta_mode=delta_mode,
        )

        # Speaker bidir cross-attention (same as v13)
        self.cross_attn_a2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_a2t       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t       = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_t2a       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a       = nn.LayerNorm(hidden_dim)
        self.speaker_drop   = nn.Dropout(dropout)

        # Fusion (same as v13)
        self.cross_attn_fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fusion       = nn.LayerNorm(hidden_dim)
        self.fuse_drop         = nn.Dropout(dropout)

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)
            h_t = self.text_encoder(text)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)
            h_t = self.text_encoder(text).unsqueeze(1)

        # Listener face: Delta CLS transformer (identity suppressed)
        listener_feat = self.video_encoder(video)

        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t     = self.norm_a2t(self.proj_a2t(a2t))
        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a     = self.norm_t2a(self.proj_t2a(t2a))

        sp_a = self.speaker_drop(a2t.mean(dim=1))
        sp_t = self.speaker_drop(t2a.mean(dim=1))

        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            sp_a = torch.zeros_like(sp_a)
            sp_t = torch.zeros_like(sp_t)

        q  = listener_feat.unsqueeze(1)
        kv = torch.stack([sp_a, sp_t], dim=1)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(self.norm_fusion(attn_out.squeeze(1) + listener_feat))

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
