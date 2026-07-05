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


class LearnableQueryPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)


class MemoCMTV5(nn.Module):
    """
    v4 + learned gate: listener attends over a2t and t2a separately (frame-level),
    then a learned per-channel gate mixes them instead of a fixed 0.5/0.5.

      attn_a = CrossAttn(Q=listener, K/V=a2t)
      attn_t = CrossAttn(Q=listener, K/V=t2a)
      g      = sigmoid(Linear([attn_a, attn_t]))          # (B, H) per-channel
      fused  = LN( g*attn_a + (1-g)*attn_t + listener_feat )

    Separate attentions keep audio/text balance guaranteed (each softmax is
    normalized within its own modality) while adding frame-level granularity.
    Everything else identical to v3.
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

        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMSeqEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        # speaker bidirectional cross-attention (same as v1/v3)
        self.cross_attn_a2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_a2t = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_t2a = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a = nn.LayerNorm(hidden_dim)

        # listener
        self.listener_pool = LearnableQueryPooling(hidden_dim, num_heads, dropout)

        # fusion: separate listener->audio and listener->text cross-attentions
        self.fuse_attn_a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.fuse_attn_t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)   # learned per-channel mix
        self.norm_fusion = nn.LayerNorm(hidden_dim)
        self.fuse_drop   = nn.Dropout(dropout)

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)
            h_t = self.text_encoder(text)
            h_v = self.video_encoder(video)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)
            h_t = self.text_encoder(text).unsqueeze(1)
            h_v = self.video_encoder(video).unsqueeze(1)

        # speaker bidir cross-attention -> keep full sequences
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t     = self.norm_a2t(self.proj_a2t(a2t))     # (B, Ta, H)
        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a     = self.norm_t2a(self.proj_t2a(t2a))     # (B, Tt, H)

        # listener
        listener_feat = self.listener_pool(h_v)          # (B, H)
        q = listener_feat.unsqueeze(1)                    # (B, 1, H)

        # fusion: listener attends over each speaker sequence SEPARATELY
        attn_a, _ = self.fuse_attn_a(query=q, key=a2t, value=a2t)   # (B, 1, H)
        attn_t, _ = self.fuse_attn_t(query=q, key=t2a, value=t2a)   # (B, 1, H)
        attn_a = attn_a.squeeze(1)                        # (B, H)
        attn_t = attn_t.squeeze(1)                        # (B, H)

        # modality dropout (training only): drop the whole speaker context
        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            attn_a = torch.zeros_like(attn_a)
            attn_t = torch.zeros_like(attn_t)

        # learned per-channel gate combine + residual visual
        g = torch.sigmoid(self.gate(torch.cat([attn_a, attn_t], dim=-1)))   # (B, H)
        features = self.fuse_drop(self.norm_fusion(g * attn_a + (1.0 - g) * attn_t + listener_feat))

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
