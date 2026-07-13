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


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE). dim = d_head."""
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        # x: (B, T, H, D_head)
        seq_len = x.shape[1]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)
        return (x * cos) + (self.rotate_half(x) * sin)


class RoPEAttentionLayer(nn.Module):
    """Pre-LN self-attention block with RoPE applied to Q and K.

    Relative-position encoding: the dot product between frame m (Q) and
    frame n (K) depends only on m - n, so the model measures temporal
    distance instead of absolute index -- robust to train/test length shift.
    """
    def __init__(self, d_model=128, nhead=4, ffn_dim=None, dropout=0.3):
        super().__init__()
        self.nhead  = nhead
        self.d_head = d_model // nhead
        ffn_dim     = ffn_dim or d_model * 4

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(dim=self.d_head)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-LN: attention over normalized input, residual on raw x
        B, T, C = x.shape
        h = self.ln1(x)
        q = self.q_proj(h).reshape(B, T, self.nhead, self.d_head)
        k = self.k_proj(h).reshape(B, T, self.nhead, self.d_head)
        v = self.v_proj(h).reshape(B, T, self.nhead, self.d_head)

        # RoPE on Q and K (relative position)
        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)   # (B, H, T, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn   = self.attn_drop(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, v)                     # (B, H, T, D)
        context = context.transpose(1, 2).reshape(B, T, C)  # (B, T, C)

        x = x + self.out_proj(context)     # residual (pre-LN)
        x = x + self.ffn(self.ln2(x))      # residual FFN (pre-LN)
        return x


class RoPECLSVideoEncoder(nn.Module):
    """
    Linear proj → prepend CLS (position 0) → stack of RoPEAttentionLayer → CLS output.
    No absolute positional encoding: RoPE handles position inside attention.
    """
    def __init__(self, in_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.cls  = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.layers = nn.ModuleList([
            RoPEAttentionLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x   = self.proj(x)                            # (B, T, H)
        cls = self.cls.expand(x.size(0), -1, -1)      # (B, 1, H)
        x   = torch.cat([cls, x], dim=1)              # (B, T+1, H) — CLS at pos 0
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.drop(x[:, 0, :])                  # (B, H) — CLS token


class MemoCMTV20(nn.Module):
    """
    v13 + Rotary Position Embedding (RoPE) for the listener (video) branch.

    Changes vs v13:
      - Video branch: nn.TransformerEncoder + fixed Sinusoidal PE
        → stack of RoPEAttentionLayer (relative position, no absolute PE).
      - CLS token sits at position 0; frames at 1..T. RoPE rotates Q/K so the
        attention score between frames depends only on their relative distance
        m - n, improving generalization across train/test length shift.
      - Speaker branch and fusion: identical to v13.

    New yaml params (with defaults):
      tf_layers: 2   number of RoPE attention layers
      tf_heads:  4   number of heads (d_head = hidden_dim / tf_heads)
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

        num_heads = max(1, hidden_dim // 64)  # speaker cross-attn / fusion

        # Speaker branches (same as v13)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)

        # Listener branch: RoPE transformer + CLS
        self.video_encoder = RoPECLSVideoEncoder(
            in_dim=video_dim, hidden_dim=hidden_dim,
            num_heads=tf_heads, num_layers=tf_layers, dropout=dropout,
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

        # Speaker encoding (sequence → sequence)
        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)    # (B, Ta, H)
            h_t = self.text_encoder(text)      # (B, Tt, H)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)
            h_t = self.text_encoder(text).unsqueeze(1)

        # Listener encoding: CLS token from RoPE Transformer
        listener_feat = self.video_encoder(video)  # (B, H)

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

        # Fusion: listener query attends 2 speaker tokens (same as v13)
        q  = listener_feat.unsqueeze(1)              # (B, 1, H)
        kv = torch.stack([sp_a, sp_t], dim=1)        # (B, 2, H)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(self.norm_fusion(attn_out.squeeze(1) + listener_feat))

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
