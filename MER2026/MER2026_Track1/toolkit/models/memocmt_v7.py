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


class MemoCMTV7(nn.Module):
    """
    v3 backbone (identical at inference) + Nonverbal Conflict Exposure (NCE)
    during training (adapted from CoRe-KD, arXiv:2605.29590).

    Rationale for MER-Cross: Individual training data always has audio/text
    consistent with the label (same person), so the model learns to TRUST
    them. At test the audio/text belong to the speaker (a different person),
    potentially carrying a different emotion than the listener target -- a
    consistency the model never saw in training. NCE manufactures that
    conflict in-batch: with probability conflict_p, a sample's audio+text are
    replaced by those of a DIFFERENT-label donor in the same batch, while the
    listener video and the label are kept. The model is thus trained to anchor
    on video (listener) and not be misled by target-inconsistent speaker cues.

    conflict_p is deliberately < 1.0: speaker audio/text carry a weak-but-real
    emotional-contagion signal (trimodal beats video-only by ~7 WAF), so full
    conflict would teach the model to discard useful context. NCE is disabled
    at eval; forward is then byte-identical to v3.
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
        self.conflict_p     = getattr(args, 'conflict_p',     0.3)

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

    def _conflict_swap(self, audio, text, emos):
        """In-batch donor swap: replace audio+text of a fraction of samples
        with those of a different-label donor. Video and label untouched.
        Returns possibly-modified (audio, text)."""
        B = audio.size(0)
        audio_src = audio.clone()          # source = pre-swap, avoids cascade
        text_src  = text.clone()
        for i in range(B):
            if torch.rand(1).item() >= self.conflict_p:
                continue
            diff = (emos != emos[i]).nonzero(as_tuple=True)[0]
            if diff.numel() == 0:          # batch is single-class for this label
                continue
            j = diff[torch.randint(diff.numel(), (1,)).item()].item()
            audio[i] = audio_src[j]        # same donor j for both -> "one speaker"
            text[i]  = text_src[j]
        return audio, text

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        # ---- Nonverbal Conflict Exposure (train-only) ----
        if self.training and self.conflict_p > 0 and 'emos' in batch:
            audio, text = self._conflict_swap(audio.clone(), text.clone(), batch['emos'])

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

        # modality dropout (training only): zero the whole speaker context
        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            sp_a = torch.zeros_like(sp_a)
            sp_t = torch.zeros_like(sp_t)

        # listener
        listener_feat = self.listener_pool(h_v)          # (B, H)

        # Fix 3: fusion over 2 speaker tokens (non-degenerate attention)
        q  = listener_feat.unsqueeze(1)                  # (B, 1, H)
        kv = torch.stack([sp_a, sp_t], dim=1)            # (B, 2, H)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(self.norm_fusion(attn_out.squeeze(1) + listener_feat))

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
