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


class VideoTransformerEncoder(nn.Module):
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
        x   = self.proj(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.pe(x)
        x   = self.encoder(x)
        return self.drop(x[:, 1:, :])              # (B, Tv, H) — frame seq only


class MemoCMTV16(nn.Module):
    """
    MBT v2: high-capacity bottleneck via flatten output.

    Changes vs v15 (mean-pool):
      - Output: bn_final.flatten(1) -> (B, N*H) instead of mean -> (B, H)
      - LayerNorm on the flattened vector before the classifier
      - Recommended config: hidden_dim=256, n_bottleneck=16 -> 4096-dim
        pre-classifier space (or hidden_dim=256, n_bottleneck=8 -> 2048-dim)

    Architecture (identical flow to v15 otherwise):
      Step 1: Bottleneck (Q) queries speaker context concat(F_a, F_t)
      Step 2: Updated bottleneck (Q) queries listener video frames F_v
      Step 3: Flatten N tokens -> LayerNorm -> classify
    """

    def __init__(self, args):
        super().__init__()
        audio_dim    = args.audio_dim
        text_dim     = args.text_dim
        video_dim    = args.video_dim
        output_dim1  = args.output_dim1
        output_dim2  = args.output_dim2
        dropout      = args.dropout
        hidden_dim   = args.hidden_dim
        self.grad_clip    = args.grad_clip
        self.feat_type    = getattr(args, 'feat_type',    'utt')
        n_bottleneck      = getattr(args, 'n_bottleneck', 16)
        tf_layers         = getattr(args, 'tf_layers',    2)
        tf_heads          = getattr(args, 'tf_heads',     max(4, hidden_dim // 32))

        num_heads = max(1, hidden_dim // 64)   # MHA heads for cross-attn layers

        # Speaker encoders (audio + text)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)

        # Listener video encoder (full frame sequence)
        self.video_encoder = VideoTransformerEncoder(
            in_dim=video_dim, hidden_dim=hidden_dim,
            num_heads=tf_heads, num_layers=tf_layers, dropout=dropout,
        )

        # Bottleneck tokens
        self.bottleneck = nn.Parameter(torch.zeros(1, n_bottleneck, hidden_dim))
        nn.init.normal_(self.bottleneck, std=0.02)

        # Cross-attn 1: bottleneck → speaker (audio+text)
        self.ca_speaker  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_speaker = nn.LayerNorm(hidden_dim)
        self.drop_speaker = nn.Dropout(dropout)

        # Cross-attn 2: updated bottleneck → listener video frames
        self.ca_video    = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_video   = nn.LayerNorm(hidden_dim)
        self.drop_video   = nn.Dropout(dropout)

        # Flatten N tokens -> (B, N*H), then classify
        flat_dim = n_bottleneck * hidden_dim
        self.norm_out  = nn.LayerNorm(flat_dim)
        self.fc_out_1  = nn.Linear(flat_dim, output_dim1)
        self.fc_out_2  = nn.Linear(flat_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        # Encode speaker modalities
        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)               # (B, Ta, H)
            h_t = self.text_encoder(text)                 # (B, Tt, H)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)  # (B, 1, H)
            h_t = self.text_encoder(text).unsqueeze(1)    # (B, 1, H)

        # Encode listener video → frame sequence
        frames_v = self.video_encoder(video)              # (B, Tv, H)

        # Speaker context: concatenate audio and text sequences
        F_s1 = torch.cat([h_a, h_t], dim=1)              # (B, Ta+Tt, H)

        # Step 1: Bottleneck queries speaker context
        B  = audio.size(0)
        bn = self.bottleneck.expand(B, -1, -1)            # (B, N, H)
        bn_s1, _ = self.ca_speaker(query=bn, key=F_s1, value=F_s1)
        bn_s1 = self.norm_speaker(self.drop_speaker(bn_s1))   # (B, N, H)

        # Step 2: Speaker-informed bottleneck queries listener video frames
        bn_final, _ = self.ca_video(query=bn_s1, key=frames_v, value=frames_v)
        bn_final = self.norm_video(self.drop_video(bn_final))  # (B, N, H)

        # Step 3: Flatten N tokens -> LayerNorm -> classify
        features  = self.norm_out(bn_final.flatten(1))    # (B, N*H)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
