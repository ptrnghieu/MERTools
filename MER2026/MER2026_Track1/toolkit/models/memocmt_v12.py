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


class MemoCMTV12(nn.Module):
    """
    v3 speaker/listener encoders, but ASYMMETRIC VIB-FiLM fusion instead of the
    symmetric 2-token cross-attention.

    Rationale (MER-Cross): the listener VIDEO is the only direct evidence of the
    listener's state; the speaker AUDIO/TEXT are merely an external stimulus with
    weak contagion value. So instead of mixing them symmetrically:

      1. VIDEO (listener) is the semantic ANCHOR (pooled listener_feat).
      2. SPEAKER audio/text are pooled (sp_a, sp_t), combined, then squeezed by a
         Variational Information Bottleneck (VIB) -> a compact code z that keeps
         only stimulus information mutually-informative with emotion, discarding
         speaker-specific detail that does not transfer across roles.
      3. z produces FiLM parameters (gamma, beta) that MODULATE the listener
         anchor: features = LN(gamma * listener_feat + beta) + listener_feat.
         FiLM layers are zero-initialised, so training starts as a pure
         video-anchor model and only learns to let the speaker context modulate
         when it helps.

    VIB KL is a train-only auxiliary (interloss = beta_vib * KL). At eval the VIB
    uses its mean (deterministic), so inference is stable.
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
        self.vib_dim        = getattr(args, 'vib_dim',        hidden_dim)
        self.beta_vib       = getattr(args, 'beta_vib',       0.01)

        num_heads = max(1, hidden_dim // 64)

        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMSeqEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        # speaker bidirectional cross-attention (same as v3)
        self.cross_attn_a2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_a2t = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_t2a = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a = nn.LayerNorm(hidden_dim)
        self.speaker_drop = nn.Dropout(dropout)

        # listener anchor
        self.listener_pool = LearnableQueryPooling(hidden_dim, num_heads, dropout)

        # speaker context -> VIB
        self.sp_proj    = nn.Linear(2 * hidden_dim, hidden_dim)
        self.vib_mu     = nn.Linear(hidden_dim, self.vib_dim)
        self.vib_logvar = nn.Linear(hidden_dim, self.vib_dim)

        # FiLM (zero-init -> starts as identity: pure video anchor)
        self.film_gamma = nn.Linear(self.vib_dim, hidden_dim)
        self.film_beta  = nn.Linear(self.vib_dim, hidden_dim)
        nn.init.zeros_(self.film_gamma.weight); nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight);  nn.init.zeros_(self.film_beta.bias)

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

        # speaker bidir cross-attention -> per-modality pool (Fix2)
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t = self.norm_a2t(self.proj_a2t(a2t))
        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a = self.norm_t2a(self.proj_t2a(t2a))
        sp_a = self.speaker_drop(a2t.mean(dim=1))
        sp_t = self.speaker_drop(t2a.mean(dim=1))
        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            sp_a = torch.zeros_like(sp_a)
            sp_t = torch.zeros_like(sp_t)

        # listener anchor
        listener_feat = self.listener_pool(h_v)              # (B, H)

        # speaker context -> VIB (compress to contagion-relevant code)
        sp = self.sp_proj(torch.cat([sp_a, sp_t], dim=-1))   # (B, H)
        mu     = self.vib_mu(sp)
        logvar = self.vib_logvar(sp)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu                                           # deterministic at eval

        # FiLM modulation of the video anchor (+ residual keeps the anchor)
        gamma = self.film_gamma(z)
        beta  = self.film_beta(z)
        film  = gamma * listener_feat + beta
        features = self.fuse_drop(self.norm_fusion(film + listener_feat))

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        interloss = torch.zeros(1, device=audio.device).squeeze()
        if self.training:
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
            interloss = self.beta_vib * kl
        return features, emos_out, vals_out, interloss
