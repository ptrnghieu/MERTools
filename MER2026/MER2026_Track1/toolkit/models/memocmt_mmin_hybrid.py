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


class MemoCMTMMINHybrid(nn.Module):
    """
    Hybrid MMIN for MER-Cross (Interlocutor domain shift).

    Keeps v1's speaker branch (real audio/text = stimulus context, even if wrong
    identity at test) AND adds an imagination branch that reconstructs the
    listener's own audio/text from their video. Fusion sees BOTH sources as
    key/value tokens and learns to weigh stimulus-context vs identity-consistent
    imagined signal.

    Speaker branch : LSTM → bidir cross-attn (a↔t) → mean pool          (real, like v1)
    Imagine branch : g_v → ĝ_a, ĝ_t (target = real pooled a/t, stop-grad) → imagined_feat
    Listener branch: LSTM → LearnableQueryPooling
    Fusion         : Q=listener, K/V=[speaker_feat, imagined_feat] + residual
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
        self.speaker_drop_p  = getattr(args, 'speaker_drop_p',  0.0)
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

        # ── Speaker: bidirectional cross-attention (real a/t) ─────────────
        self.cross_attn_a2t = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.proj_a2t = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t = nn.LayerNorm(hidden_dim)

        self.cross_attn_t2a = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.proj_t2a = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a = nn.LayerNorm(hidden_dim)

        self.speaker_drop = nn.Dropout(dropout)

        # ── Imagination: video → audio / text ─────────────────────────────
        self.imagine_a = ImagineMLP(hidden_dim, dropout)
        self.imagine_t = ImagineMLP(hidden_dim, dropout)
        self.im_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.im_norm   = nn.LayerNorm(hidden_dim)

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
            h_a = self.audio_encoder(audio).unsqueeze(1)
            h_t = self.text_encoder(text).unsqueeze(1)
            h_v = self.video_encoder(video).unsqueeze(1)

        # ── Speaker: bidir cross-attention → mean pool (real context) ─────
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t     = self.norm_a2t(self.proj_a2t(a2t))
        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a     = self.norm_t2a(self.proj_t2a(t2a))
        speaker_seq  = torch.cat([a2t, t2a], dim=1)
        speaker_feat = self.speaker_drop(speaker_seq.mean(dim=1))   # (B, H)

        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            speaker_feat = torch.zeros_like(speaker_feat)

        # ── Imagine listener's own audio/text from video ──────────────────
        g_a = h_a.mean(dim=1)   # real audio  — target only
        g_t = h_t.mean(dim=1)   # real text   — target only
        g_v = h_v.mean(dim=1)   # video       — imagination input
        ga_hat = self.imagine_a(g_v)
        gt_hat = self.imagine_t(g_v)
        imagined_feat = self.im_norm(self.im_proj((ga_hat + gt_hat) * 0.5))  # (B, H)

        # ── Listener: learnable query pooling ─────────────────────────────
        listener_feat = self.listener_pool(h_v)                # (B, H)

        # ── Fusion: Q=listener, K/V=[speaker, imagined] + residual ────────
        q  = listener_feat.unsqueeze(1)                        # (B, 1, H)
        kv = torch.stack([speaker_feat, imagined_feat], dim=1) # (B, 2, H)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(
            self.norm_fusion(attn_out.squeeze(1) + listener_feat)
        )                                                       # (B, H)

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        # ── Imagination loss (stop-grad targets) ──────────────────────────
        l_forward = F.mse_loss(ga_hat, g_a.detach()) + F.mse_loss(gt_hat, g_t.detach())
        interloss = self.lambda_forward * l_forward

        return features, emos_out, vals_out, interloss
