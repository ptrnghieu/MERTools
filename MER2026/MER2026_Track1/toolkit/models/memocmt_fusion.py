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
        q = self.query.expand(x.size(0), -1, -1)    # (B, 1, H)
        out, _ = self.attn(query=q, key=x, value=x)  # (B, 1, H)
        return out.squeeze(1)                         # (B, H)


class CmIRHead(nn.Module):
    """Project hidden → (Z_inv, Z_spu). Works on (B, H) or (B, T, H)."""

    def __init__(self, in_dim, inv_dim):
        super().__init__()
        self.proj_inv = nn.Linear(in_dim, inv_dim)
        self.proj_spu = nn.Linear(in_dim, inv_dim)

    def forward(self, x):
        return self.proj_inv(x), self.proj_spu(x)


class CmIRDecoder(nn.Module):
    """Reconstruct input from cat(Z_inv, Z_spu) — the R_rec anchor."""

    def __init__(self, inv_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(inv_dim * 2, out_dim)

    def forward(self, z_inv, z_spu):
        return self.fc(torch.cat([z_inv, z_spu], dim=-1))


class MemoCMTFusion(nn.Module):
    """
    Speaker branch : LSTM → bidir cross-attention (full T) → CmIR Head
                     → Z_inv_s (B, T_a+T_t, H//2)
    Listener branch: LSTM → LearnableQueryPooling → CmIR Head
                     → Z_inv_v (B, H//2)
    Fusion         : Q=Z_inv_v (B,1,H//2), K/V=Z_inv_s (B,T,H//2)
                     → cross-attention + residual → classifier
    CmIR losses    : R_inv + R_dec + R_rec for both branches (training only)
    """

    def __init__(self, args):
        super().__init__()

        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        dropout     = args.dropout
        hidden_dim  = args.hidden_dim
        self.grad_clip = args.grad_clip

        self.feat_type       = getattr(args, 'feat_type', 'utt')
        self.speaker_drop_p  = getattr(args, 'speaker_drop_p', 0.2)

        # CmIR loss weights — small defaults so task loss dominates
        self.lambda_inv  = getattr(args, 'lambda_inv',  0.05)
        self.lambda_dec  = getattr(args, 'lambda_dec',  0.05)
        self.lambda_rec  = getattr(args, 'lambda_rec',  0.05)
        self.alpha_dec   = getattr(args, 'alpha_dec',   0.1)   # off-diagonal penalty
        self.noise_std   = getattr(args, 'noise_std',   0.1)   # virtual environment noise

        inv_dim       = hidden_dim // 2
        speaker_heads = max(1, hidden_dim // 64)
        fusion_heads  = max(1, inv_dim // 64)

        # ── Encoders ───────────────────────────────────────────────────────
        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMSeqEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        # ── Speaker: bidir cross-attention on full sequence ────────────────
        self.cross_attn_a2t = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=speaker_heads,
            dropout=dropout, batch_first=True,
        )
        self.proj_a2t = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t = nn.LayerNorm(hidden_dim)

        self.cross_attn_t2a = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=speaker_heads,
            dropout=dropout, batch_first=True,
        )
        self.proj_t2a = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a = nn.LayerNorm(hidden_dim)

        self.speaker_drop = nn.Dropout(dropout)

        # ── Listener: learnable query pooling ─────────────────────────────
        self.listener_pool = LearnableQueryPooling(hidden_dim, speaker_heads, dropout)

        # ── CmIR Heads (Linear on last dim — transparent to sequence dim) ─
        self.cmir_speaker  = CmIRHead(hidden_dim, inv_dim)
        self.cmir_listener = CmIRHead(hidden_dim, inv_dim)

        # ── CmIR Decoders for R_rec ────────────────────────────────────────
        self.dec_speaker  = CmIRDecoder(inv_dim, hidden_dim)
        self.dec_listener = CmIRDecoder(inv_dim, hidden_dim)

        # ── Fusion: Q=(B,1,H//2), K/V=(B,T,H//2) — Softmax over T tokens ──
        self.cross_attn_fusion = nn.MultiheadAttention(
            embed_dim=inv_dim, num_heads=fusion_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_fusion = nn.LayerNorm(inv_dim)
        self.fuse_drop   = nn.Dropout(dropout)

        # ── Classifier (valence head removed — output_dim2=0 for MER2026) ──
        self.fc_out_1 = nn.Linear(inv_dim, output_dim1)

    def _cmir_losses(self, x_speaker, x_listener):
        """
        CmIR auxiliary losses for speaker and listener modalities.

        x_speaker:  (B, H) — mean-pooled speaker_seq
        x_listener: (B, H) — h_v_pooled

        Returns a scalar loss.
        """
        total = x_speaker.new_zeros(1).squeeze()

        for x, head, decoder in (
            (x_speaker,  self.cmir_speaker,  self.dec_speaker),
            (x_listener, self.cmir_listener, self.dec_listener),
        ):
            # R_inv: two independently noisy views → Z_inv must be stable
            z_inv_e1, _ = head(x + torch.randn_like(x) * self.noise_std)
            z_inv_e2, _ = head(x + torch.randn_like(x) * self.noise_std)
            r_inv = (z_inv_e1 - z_inv_e2).abs().mean()

            # R_dec: sample-wise correlation C^m = Z_inv @ Z_spu.T → (B, B)
            # Penalise diagonal (same-sample) + alpha * off-diagonal (cross-sample)
            z_inv, z_spu = head(x)
            z_inv_n = F.normalize(z_inv, dim=1)   # (B, H//2)
            z_spu_n = F.normalize(z_spu, dim=1)
            C      = torch.matmul(z_inv_n, z_spu_n.T)   # (B, B)
            diag_C = torch.diag(torch.diag(C))
            off_C  = C - diag_C
            r_dec  = (diag_C + self.alpha_dec * off_C).norm()

            # R_rec: Linear decoder reconstructs x — prevents Z=0 degenerate solution
            x_hat = decoder(z_inv, z_spu)        # (B, H)
            r_rec = F.mse_loss(x_hat, x.detach())

            total = total + (
                self.lambda_inv * r_inv
                + self.lambda_dec * r_dec
                + self.lambda_rec * r_rec
            )

        return total

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

        # ── Speaker: bidir cross-attention — keep FULL SEQUENCE ───────────
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t     = self.norm_a2t(self.proj_a2t(a2t) + h_a)   # (B, T_a, H) + residual

        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a     = self.norm_t2a(self.proj_t2a(t2a) + h_t)   # (B, T_t, H) + residual

        speaker_seq = torch.cat([a2t, t2a], dim=1)           # (B, T_a+T_t, H)

        # ── Listener: pool to utterance vector ────────────────────────────
        h_v_pooled = self.listener_pool(h_v)                  # (B, H)

        # ── CmIR auxiliary losses (training only) ─────────────────────────
        interloss = audio.new_zeros(1).squeeze()
        if self.training:
            x_spk    = self.speaker_drop(speaker_seq.mean(dim=1))  # (B, H) for loss only
            interloss = self._cmir_losses(x_spk, h_v_pooled)

        # ── CmIR Heads — only Z_inv flows downstream ───────────────────────
        # Linear on last dim works transparently on (B, T, H) → (B, T, H//2)
        Z_inv_s, _ = self.cmir_speaker(speaker_seq)    # (B, T_a+T_t, H//2)
        Z_inv_v, _ = self.cmir_listener(h_v_pooled)    # (B, H//2)

        # ── Speaker modality dropout ───────────────────────────────────────
        if self.training and torch.rand(1).item() < self.speaker_drop_p:
            Z_inv_s = torch.zeros_like(Z_inv_s)

        # ── Fusion: Softmax is meaningful — K/V has T_a+T_t > 1 tokens ────
        q = Z_inv_v.unsqueeze(1)                              # (B, 1, H//2)
        attn_out, _ = self.cross_attn_fusion(
            query=q, key=Z_inv_s, value=Z_inv_s,
        )                                                      # (B, 1, H//2)
        features = self.fuse_drop(
            self.norm_fusion(attn_out.squeeze(1) + Z_inv_v)
        )                                                      # (B, H//2)

        emos_out = self.fc_out_1(features)
        vals_out = torch.zeros(features.size(0), 1, device=audio.device)

        return features, emos_out, vals_out, interloss
