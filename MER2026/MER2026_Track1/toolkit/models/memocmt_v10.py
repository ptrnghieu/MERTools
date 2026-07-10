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


_AU_ORDER = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
             'AU11', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24',
             'AU25', 'AU26', 'AU28', 'AU43']
_MOUTH_SPEECH = ['AU10', 'AU14', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28']
_MOUTH_IDX = [_AU_ORDER.index(a) for a in _MOUTH_SPEECH]


class MemoCMTV10(nn.Module):
    """
    v3 with a NON-DISRUPTIVE AU add-on. The CLIP video anchor is kept EXACTLY
    as in v3 (listener_feat = pooled CLIP, used as the residual anchor of the
    fusion). The AU branch produces au_feat, added as a THIRD key/value token
    in the fusion (kv = [sp_a, sp_t, au_feat]) that the attention may use or
    ignore. So AU can only ADD: if it is noise, attention downweights it and
    the score stays ~v3; the strong CLIP path can no longer be dragged down
    the way v9's anchor-replacement fusion did.

    Consumes clip_au20-FRA (768 CLIP + 20 AU); splits inside the model.
    au_mouth_drop zeros speech/articulation mouth AUs.
    """

    def __init__(self, args):
        super().__init__()
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim               # 788 = 768 clip + 20 au
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout     = args.dropout
        hidden_dim  = args.hidden_dim
        self.grad_clip = args.grad_clip
        self.feat_type      = getattr(args, 'feat_type',      'utt')
        self.speaker_drop_p = getattr(args, 'speaker_drop_p', 0.0)
        self.clip_split     = getattr(args, 'clip_split',     768)
        self.au_mouth_drop  = getattr(args, 'au_mouth_drop',  False)
        au_dim = video_dim - self.clip_split
        assert au_dim > 0
        self.register_buffer('mouth_idx', torch.tensor(_MOUTH_IDX, dtype=torch.long))

        num_heads = max(1, hidden_dim // 64)
        seq = self.feat_type in ['frm_align', 'frm_unalign']
        Enc = (lambda d: LSTMSeqEncoder(d, hidden_dim, dropout)) if seq \
              else (lambda d: MLPEncoder(d, hidden_dim, dropout))

        self.audio_encoder = Enc(audio_dim)
        self.text_encoder  = Enc(text_dim)
        self.clip_encoder  = Enc(self.clip_split)   # CLIP video anchor (v3's video path)
        self.au_encoder    = Enc(au_dim)            # AU add-on branch

        self.listener_pool = LearnableQueryPooling(hidden_dim, num_heads, dropout)
        self.au_pool       = LearnableQueryPooling(hidden_dim, num_heads, dropout)

        # speaker branch (v3)
        self.cross_attn_a2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_a2t = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_t2a = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a = nn.LayerNorm(hidden_dim)
        self.speaker_drop = nn.Dropout(dropout)

        # fusion: K/V = [sp_a, sp_t, au_feat] (3 tokens); Q = CLIP listener_feat
        self.cross_attn_fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fusion = nn.LayerNorm(hidden_dim)
        self.fuse_drop   = nn.Dropout(dropout)

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def _enc(self, enc, x):
        if self.feat_type in ['frm_align', 'frm_unalign']:
            return enc(x)
        return enc(x).unsqueeze(1)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']                        # (B, T, 788)

        clip_part = video[..., :self.clip_split]
        au_part   = video[..., self.clip_split:]
        if self.au_mouth_drop:
            au_part = au_part.clone()
            au_part[..., self.mouth_idx] = 0.0

        h_a = self._enc(self.audio_encoder, audio)
        h_t = self._enc(self.text_encoder,  text)
        h_v = self._enc(self.clip_encoder,  clip_part)
        h_au = self._enc(self.au_encoder,   au_part)

        # speaker branch (v3)
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t = self.norm_a2t(self.proj_a2t(a2t))
        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a = self.norm_t2a(self.proj_t2a(t2a))
        sp_a = self.speaker_drop(a2t.mean(dim=1))
        sp_t = self.speaker_drop(t2a.mean(dim=1))
        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            sp_a = torch.zeros_like(sp_a)
            sp_t = torch.zeros_like(sp_t)

        # CLIP anchor (v3) + AU as an extra fusion token
        listener_feat = self.listener_pool(h_v)        # pure CLIP (B, H)  -- anchor
        au_feat = self.au_pool(h_au)                   # (B, H)

        q  = listener_feat.unsqueeze(1)                # (B, 1, H)
        kv = torch.stack([sp_a, sp_t, au_feat], dim=1) # (B, 3, H)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(self.norm_fusion(attn_out.squeeze(1) + listener_feat))

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
