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


class BagOfFramesTransformerEncoder(nn.Module):
    """
    Linear proj → prepend CLS → TransformerEncoder (NO positional encoding).

    Without PE the transformer treats frames as an unordered set — CLS
    accumulates frequency/intensity of emotional states rather than learning
    a linear temporal narrative.  Motivated by the train/test role shift:
    train video belongs to the speaker (temporal order meaningful); test video
    belongs to the listener (set of expressions, order less informative).
    """
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.cls  = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
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
        x   = torch.cat([cls, x], dim=1)              # (B, T+1, H) — NO PE
        x   = self.encoder(x)                         # (B, T+1, H)
        return self.drop(x[:, 0, :])                  # (B, H) — CLS output


class MemoCMTV17(nn.Module):
    """
    v13 + Bag-of-Frames: identical to v13 but NO positional encoding on video.

    Hypothesis: train video is the speaker's own face (temporal order is a
    meaningful signal), but test video is the listener's face (what matters is
    WHICH expressions appear and how often, not their order).  Removing PE
    turns the transformer into a permutation-invariant set aggregator,
    potentially reducing overfitting to train-specific temporal patterns.
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

        # Speaker branches (same as v3/v13)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)

        # Listener branch: Bag-of-Frames (no PE)
        self.video_encoder = BagOfFramesTransformerEncoder(
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

        # Fusion: same 2-token cross-attention as v3/v13
        self.cross_attn_fuse = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_fuse       = nn.Linear(hidden_dim, hidden_dim)
        self.norm_fuse       = nn.LayerNorm(hidden_dim)

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

        # Listener: Bag-of-Frames CLS (no PE)
        listener_feat = self.video_encoder(video)  # (B, H)

        # Speaker bidir cross-attention
        sp_a, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        sp_a     = self.norm_a2t(self.proj_a2t(sp_a))   # (B, Ta, H)
        sp_t, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        sp_t     = self.norm_t2a(self.proj_t2a(sp_t))   # (B, Tt, H)

        # Optional: zero out entire speaker context during training
        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            sp_a = torch.zeros_like(sp_a)
            sp_t = torch.zeros_like(sp_t)
        else:
            sp_a = self.speaker_drop(sp_a)
            sp_t = self.speaker_drop(sp_t)

        # Mean-pool speaker sequences → 2 tokens
        sp_a_tok = sp_a.mean(dim=1, keepdim=True)   # (B, 1, H)
        sp_t_tok = sp_t.mean(dim=1, keepdim=True)   # (B, 1, H)
        sp_tokens = torch.cat([sp_a_tok, sp_t_tok], dim=1)  # (B, 2, H)

        # Fusion: listener CLS queries 2 speaker tokens
        query = listener_feat.unsqueeze(1)           # (B, 1, H)
        fused, _ = self.cross_attn_fuse(query=query, key=sp_tokens, value=sp_tokens)
        features  = self.norm_fuse(self.proj_fuse(fused).squeeze(1) + listener_feat)  # (B, H)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
