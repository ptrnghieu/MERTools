import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder


class LSTMSeqEncoder(nn.Module):
    """LSTM encoder that returns the full sequence (B, T, hidden) instead of final state."""

    def __init__(self, in_size, hidden_size, dropout, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers,
                           batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, in_size)
        out, _ = self.rnn(x)       # out: (B, T, hidden_size)
        return self.dropout(out)


class SpeakerListenerFusion(nn.Module):
    """
    Two separate encoders for speaker (audio+text) and listener (video),
    fused via cross-attention: listener video = Q, speaker audio+text = K, V.

    - UTT features: single vector per sample → cross-attention over 2 speaker tokens
    - FRA features: full sequence → cross-attention preserves temporal info
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

        feat_type = getattr(args, 'feat_type', 'utt')
        self.feat_type = feat_type

        if feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMSeqEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        num_heads = max(1, hidden_dim // 64)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        h_a = self.audio_encoder(audio)   # (B, T_a, hidden) or (B, hidden)
        h_t = self.text_encoder(text)     # (B, T_t, hidden) or (B, hidden)
        h_v = self.video_encoder(video)   # (B, T_v, hidden) or (B, hidden)

        if self.feat_type in ['frm_align', 'frm_unalign']:
            # Full sequences: cat audio and text along time dim as speaker context
            speaker_kv = torch.cat([h_a, h_t], dim=1)  # (B, T_a+T_t, hidden)
            query      = h_v                             # (B, T_v, hidden)
        else:
            # Single vectors: add seq dim
            speaker_kv = torch.cat([h_a.unsqueeze(1), h_t.unsqueeze(1)], dim=1)  # (B, 2, hidden)
            query      = h_v.unsqueeze(1)                                          # (B, 1, hidden)

        # Cross-attention: listener video queries speaker audio+text
        attn_out, _ = self.cross_attn(
            query=query,
            key=speaker_kv,
            value=speaker_kv,
        )  # same shape as query

        # Pool over time
        features = attn_out.mean(dim=1)  # (B, hidden)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()

        return features, emos_out, vals_out, interloss
