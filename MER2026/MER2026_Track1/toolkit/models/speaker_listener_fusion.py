import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder, LSTMEncoder


class SpeakerListenerFusion(nn.Module):
    """
    Two separate encoders for speaker (audio+text) and listener (video),
    fused via cross-attention: listener video = Q, speaker audio+text = K, V.
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
        if feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)
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

        # Encode: all outputs are (B, hidden_dim)
        h_a = self.audio_encoder(audio)
        h_t = self.text_encoder(text)
        h_v = self.video_encoder(video)

        # Add sequence dimension for MultiheadAttention
        h_a = h_a.unsqueeze(1)   # (B, 1, hidden_dim)
        h_t = h_t.unsqueeze(1)   # (B, 1, hidden_dim)
        h_v = h_v.unsqueeze(1)   # (B, 1, hidden_dim)  — Q

        # Speaker context: audio + text as 2 tokens
        speaker_kv = torch.cat([h_a, h_t], dim=1)  # (B, 2, hidden_dim)

        # Cross-attention: listener video queries speaker audio+text
        attn_out, _ = self.cross_attn(
            query=h_v,
            key=speaker_kv,
            value=speaker_kv,
        )  # (B, 1, hidden_dim)

        features = attn_out.squeeze(1)  # (B, hidden_dim)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()

        return features, emos_out, vals_out, interloss
