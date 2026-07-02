import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """Single learnable query attends over sequence → (B, H)."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)     # (B, 1, H)
        out, _ = self.attn(query=q, key=x, value=x)  # (B, 1, H)
        return out.squeeze(1)                         # (B, H)


class ImagineMLP(nn.Module):
    """Cross-modal imagination: video embedding → (audio|text) embedding."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class MemoCMTMMIN(nn.Module):
    """
    Bare MMIN for MER-Cross (Interlocutor domain shift).

    Insight: at test only the listener VIDEO is trustworthy (audio/text belong to
    the speaker = wrong identity). Train (Individual) has a/t/v from the same
    person, giving a genuine target to learn "video → imagine this person's a/t".

    Listener branch  : LSTM → LearnableQueryPooling
    Imagination      : g_v → ĝ_a, g_v → ĝ_t   (MLP, target = real pooled a/t, stop-grad)
    Fusion           : Q=listener, K/V=imagined speaker + residual
    Test path uses ONLY imagined a/t; speaker's real a/t serves only as a loss target.
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

        self.feat_type       = getattr(args, 'feat_type',       'utt')
        self.lambda_forward  = getattr(args, 'lambda_forward',  1.0)

        num_heads = max(1, hidden_dim // 64)

        # ── Encoders ──────────────────────────────────────────────────────
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMSeqEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        # ── Imagination: video → audio / text ─────────────────────────────
        self.imagine_a = ImagineMLP(hidden_dim, dropout)
        self.imagine_t = ImagineMLP(hidden_dim, dropout)

        # ── Imagined speaker vector ───────────────────────────────────────
        self.sp_proj = nn.Linear(hidden_dim, hidden_dim)
        self.sp_norm = nn.LayerNorm(hidden_dim)

        # ── Listener: learnable query pooling ─────────────────────────────
        self.listener_pool = LearnableQueryPooling(hidden_dim, num_heads, dropout)

        # ── Fusion ────────────────────────────────────────────────────────
        self.cross_attn_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_fusion = nn.LayerNorm(hidden_dim)
        self.fuse_drop   = nn.Dropout(dropout)

        # ── Classifier ────────────────────────────────────────────────────
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        # ── Encode ────────────────────────────────────────────────────────
        if self.feat_type in ['frm_align', 'frm_unalign']:
            h_a = self.audio_encoder(audio)   # (B, T_a, H)
            h_t = self.text_encoder(text)     # (B, T_t, H)
            h_v = self.video_encoder(video)   # (B, T_v, H)
        else:
            h_a = self.audio_encoder(audio).unsqueeze(1)  # (B, 1, H)
            h_t = self.text_encoder(text).unsqueeze(1)    # (B, 1, H)
            h_v = self.video_encoder(video).unsqueeze(1)  # (B, 1, H)

        # ── Pooled vectors ────────────────────────────────────────────────
        g_a = h_a.mean(dim=1)   # (B, H) real audio  — imagination TARGET only
        g_t = h_t.mean(dim=1)   # (B, H) real text   — imagination TARGET only
        g_v = h_v.mean(dim=1)   # (B, H) video       — imagination INPUT

        # ── Imagine listener's own audio/text from video ──────────────────
        ga_hat = self.imagine_a(g_v)   # (B, H)
        gt_hat = self.imagine_t(g_v)   # (B, H)

        # ── Imagined "speaker" vector (never uses real a/t in forward path) ─
        sp = self.sp_norm(self.sp_proj((ga_hat + gt_hat) * 0.5))   # (B, H)

        # ── Listener: learnable query pooling ─────────────────────────────
        listener_feat = self.listener_pool(h_v)                # (B, H)

        # ── Fusion: Q=listener, K/V=imagined speaker + residual ───────────
        q  = listener_feat.unsqueeze(1)   # (B, 1, H)
        kv = sp.unsqueeze(1)              # (B, 1, H)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(
            self.norm_fusion(attn_out.squeeze(1) + listener_feat)
        )                                                       # (B, H)

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        # ── Imagination loss (stop-grad on real targets to avoid collapse) ─
        l_forward = F.mse_loss(ga_hat, g_a.detach()) + F.mse_loss(gt_hat, g_t.detach())
        interloss = self.lambda_forward * l_forward

        return features, emos_out, vals_out, interloss
