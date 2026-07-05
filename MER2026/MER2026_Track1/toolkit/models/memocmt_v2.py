import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder


class LSTMSeqEncoder(nn.Module):
    """Unidirectional LSTM (for audio/text)."""
    def __init__(self, in_size, hidden_size, dropout):
        super().__init__()
        self.input_drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(in_size, hidden_size, batch_first=True, bidirectional=False)
        self.output_drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.rnn(self.input_drop(x))
        return self.output_drop(out)


class BiLSTMSeqEncoder(nn.Module):
    """Bidirectional LSTM + projection back to hidden (for visual)."""
    def __init__(self, in_size, hidden_size, dropout):
        super().__init__()
        self.input_drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(in_size, hidden_size, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden_size, hidden_size)
        self.output_drop = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.rnn(self.input_drop(x))     # (B, T, 2H)
        return self.output_drop(self.proj(out))    # (B, T, H)


class SelfBlock(nn.Module):
    """Transformer self-attention block."""
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.drop(self.attn(x, x, x)[0]))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class CrossBlock(nn.Module):
    """Transformer cross-attention block (query attends to key/value)."""
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv):
        x = self.norm1(q + self.drop(self.attn(q, kv, kv)[0]))   # residual on query (visual)
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


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


class MemoCMTV2(nn.Module):
    """
    Visual-anchored fusion for MER-Cross (video is the trustworthy signal at test).

    Visual branch : BiLSTM → self-attention → attends to audio & text (visual=query)
    Gated fusion  : g*v_a + (1-g)*v_t + residual(h_v)   ← raw visual survives when a/t unreliable
    Pool          : LearnableQueryPooling over the enriched visual sequence
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
        self.is_seq = self.feat_type in ['frm_align', 'frm_unalign']

        if self.is_seq:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = BiLSTMSeqEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        self.visual_self = SelfBlock(hidden_dim, num_heads, dropout)
        self.cross_va = CrossBlock(hidden_dim, num_heads, dropout)   # visual ← audio
        self.cross_vt = CrossBlock(hidden_dim, num_heads, dropout)   # visual ← text

        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm_enr = nn.LayerNorm(hidden_dim)

        self.pool = LearnableQueryPooling(hidden_dim, num_heads, dropout)
        self.out_drop = nn.Dropout(dropout)

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        if self.is_seq:
            h_a = self.audio_encoder(audio)   # (B, Ta, H)
            h_t = self.text_encoder(text)     # (B, Tt, H)
            h_v = self.video_encoder(video)   # (B, Tv, H)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)
            h_t = self.text_encoder(text).unsqueeze(1)
            h_v = self.video_encoder(video).unsqueeze(1)

        # visual temporal self-attention
        h_v = self.visual_self(h_v)

        # visual attends to audio & text (visual = query)
        v_a = self.cross_va(h_v, h_a)         # (B, Tv, H)
        v_t = self.cross_vt(h_v, h_t)         # (B, Tv, H)

        # gated fusion + raw-visual residual
        g = torch.sigmoid(self.gate(torch.cat([v_a, v_t], dim=-1)))   # (B, Tv, H)
        v_enr = self.norm_enr(g * v_a + (1.0 - g) * v_t + h_v)         # (B, Tv, H)

        # pool enriched visual
        feat = self.out_drop(self.pool(v_enr))                        # (B, H)

        emos_out = self.fc_out_1(feat)
        vals_out = self.fc_out_2(feat)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return feat, emos_out, vals_out, interloss
