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
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)


class TranslationHead(nn.Module):
    """Small MLP that translates the listener (video) feature into a speaker
    modality representation. Train-only; discarded at inference."""
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class MemoCMTV6(nn.Module):
    """
    v3 backbone (unchanged at inference) + CM-StEW-style auxiliary losses
    during training (Rajan et al., arXiv:2108.00809):

      - AUX-1 translation: MLP heads map listener_feat -> sp_a.detach(),
        sp_t.detach() with L1 loss. Teaches the video encoder to carry
        information predictive of the same person's vocal/lexical affect.
      - AUX-2 alignment:   1 - cos(listener_feat, mean(sp_a, sp_t).detach()),
        a stable stand-in for the paper's DCCA correlation alignment.

    Targets are detached so auxiliary gradients flow ONLY into the video
    branch (+ translation heads): the speaker branch stays a fixed teacher
    w.r.t. knowledge transfer and is trained by CE alone.

    Rationale for MER-Cross: at test time video is the only same-identity
    signal (listener), while audio/text belong to the speaker. In Individual
    training data all three modalities share one identity, so audio/text can
    legitimately teach the video encoder there; at test the enriched video
    branch is what transfers. interloss = alpha*L_align + beta*L_trans is
    returned only in training mode (zero at eval).
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
        self.alpha_align    = getattr(args, 'alpha_align',    0.3)
        self.beta_trans     = getattr(args, 'beta_trans',     0.3)

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
        self.speaker_drop = nn.Dropout(dropout)

        # listener
        self.listener_pool = LearnableQueryPooling(hidden_dim, num_heads, dropout)

        # fusion (K/V = 2 speaker tokens) -- identical to v3
        self.cross_attn_fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fusion = nn.LayerNorm(hidden_dim)
        self.fuse_drop   = nn.Dropout(dropout)

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

        # CM-StEW auxiliary heads (train-only)
        self.trans_a = TranslationHead(hidden_dim, dropout)
        self.trans_t = TranslationHead(hidden_dim, dropout)

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

        # speaker bidir cross-attention
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t     = self.norm_a2t(self.proj_a2t(a2t))     # (B, Ta, H)
        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a     = self.norm_t2a(self.proj_t2a(t2a))     # (B, Tt, H)

        # Fix 2: pool each modality separately -> equal weight
        sp_a = self.speaker_drop(a2t.mean(dim=1))       # (B, H)
        sp_t = self.speaker_drop(t2a.mean(dim=1))       # (B, H)

        # listener
        listener_feat = self.listener_pool(h_v)          # (B, H)

        # ---- CM-StEW auxiliary losses (train-only, targets detached) ----
        # computed BEFORE speaker modality dropout so teachers are never zeroed
        interloss = torch.zeros(1, device=audio.device).squeeze()
        if self.training:
            tgt_a = sp_a.detach()
            tgt_t = sp_t.detach()
            # AUX-1: translation video -> audio / text representations
            loss_trans = F.l1_loss(self.trans_a(listener_feat), tgt_a) \
                       + F.l1_loss(self.trans_t(listener_feat), tgt_t)
            # AUX-2: alignment video <-> mean speaker representation
            tgt_s = 0.5 * (tgt_a + tgt_t)
            loss_align = (1.0 - F.cosine_similarity(listener_feat, tgt_s, dim=-1)).mean()
            interloss = self.alpha_align * loss_align + self.beta_trans * loss_trans

        # modality dropout (training only): zero the whole speaker context
        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            sp_a = torch.zeros_like(sp_a)
            sp_t = torch.zeros_like(sp_t)

        # Fix 3: fusion over 2 speaker tokens (non-degenerate attention)
        q  = listener_feat.unsqueeze(1)                  # (B, 1, H)
        kv = torch.stack([sp_a, sp_t], dim=1)            # (B, 2, H)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(self.norm_fusion(attn_out.squeeze(1) + listener_feat))

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        return features, emos_out, vals_out, interloss
