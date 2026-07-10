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


# AU column order produced by extract_au.py (py-feat 20 AUs), within the AU
# sub-block [0:20] of the au part. Speech/articulation-heavy mouth AUs:
_AU_ORDER = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10',
             'AU11', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24',
             'AU25', 'AU26', 'AU28', 'AU43']
_MOUTH_SPEECH = ['AU10', 'AU14', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28']
_MOUTH_IDX = [_AU_ORDER.index(a) for a in _MOUTH_SPEECH]   # [7,10,12,13,14,15,16,17,18]


class MemoCMTV9(nn.Module):
    """
    v3 speaker/fusion, but the video side is split into two dedicated branches
    fused at embedding level:
      - CLIP branch: semantic face appearance (first `clip_split` dims)
      - AU branch:   FACS action units + normalized landmarks + head pose
                     (remaining dims), identity-invariant expression dynamics
    Gated embedding fusion -> listener_feat, then v3's speaker cross-attn +
    2-token fusion unchanged.

    Consumes the pre-merged clip_au-FRA feature (--video_feature=clip_au-FRA)
    and splits it back inside the model, so the dataloader is untouched.

    au_mouth_drop=True zeros the speech/articulation-heavy mouth AUs to probe
    the mouth-AU speech-contamination hypothesis (train speakers talk -> mouth
    AUs encode articulation; test listeners are silent -> same AUs encode
    emotion).
    """

    def __init__(self, args):
        super().__init__()
        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim               # e.g. 930 = 768 clip + 162 au
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout     = args.dropout
        hidden_dim  = args.hidden_dim
        self.grad_clip = args.grad_clip
        self.feat_type      = getattr(args, 'feat_type',      'utt')
        self.speaker_drop_p = getattr(args, 'speaker_drop_p', 0.0)
        self.clip_split     = getattr(args, 'clip_split',     768)
        self.au_mouth_drop  = getattr(args, 'au_mouth_drop',  False)
        self.au_gate        = getattr(args, 'au_gate',        False)
        au_dim = video_dim - self.clip_split
        assert au_dim > 0, f'video_dim {video_dim} <= clip_split {self.clip_split}'
        self.register_buffer('mouth_idx', torch.tensor(_MOUTH_IDX, dtype=torch.long))

        num_heads = max(1, hidden_dim // 64)
        seq = self.feat_type in ['frm_align', 'frm_unalign']

        Enc = (lambda d: LSTMSeqEncoder(d, hidden_dim, dropout)) if seq \
              else (lambda d: MLPEncoder(d, hidden_dim, dropout))
        self.audio_encoder = Enc(audio_dim)
        self.text_encoder  = Enc(text_dim)
        self.clip_encoder  = Enc(self.clip_split)
        self.au_encoder    = Enc(au_dim)

        # video-side pooling + gated embedding fusion
        self.clip_pool = LearnableQueryPooling(hidden_dim, num_heads, dropout)
        self.au_pool   = LearnableQueryPooling(hidden_dim, num_heads, dropout)
        # default: concat -> proj (both contribute independently, no zero-sum
        # competition that would let strong CLIP suppress the weaker new AU).
        # au_gate=True switches to an elementwise gated convex mix (experiment).
        self.vproj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.vgate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

        # speaker branch (identical to v3)
        self.cross_attn_a2t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_a2t = nn.Linear(hidden_dim, hidden_dim)
        self.norm_a2t = nn.LayerNorm(hidden_dim)
        self.cross_attn_t2a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.proj_t2a = nn.Linear(hidden_dim, hidden_dim)
        self.norm_t2a = nn.LayerNorm(hidden_dim)
        self.speaker_drop = nn.Dropout(dropout)

        # fusion (v3 Fix3)
        self.cross_attn_fusion = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_fusion = nn.LayerNorm(hidden_dim)
        self.fuse_drop   = nn.Dropout(dropout)

        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def _enc(self, encoder, x):
        if self.feat_type in ['frm_align', 'frm_unalign']:
            return encoder(x)
        return encoder(x).unsqueeze(1)

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']                       # (B, T, 930)

        clip_part = video[..., :self.clip_split]
        au_part   = video[..., self.clip_split:]
        if self.au_mouth_drop:
            au_part = au_part.clone()
            au_part[..., self.mouth_idx] = 0.0        # AU block is at [0:20] of au_part

        h_a = self._enc(self.audio_encoder, audio)
        h_t = self._enc(self.text_encoder,  text)
        h_clip = self._enc(self.clip_encoder, clip_part)
        h_au   = self._enc(self.au_encoder,   au_part)

        # video-side gated embedding fusion
        clip_feat = self.clip_pool(h_clip)            # (B, H)
        au_feat   = self.au_pool(h_au)                # (B, H)
        cat = torch.cat([clip_feat, au_feat], dim=-1)
        if self.au_gate:
            g = torch.sigmoid(self.vgate(cat))        # elementwise gate
            fused = g * clip_feat + (1.0 - g) * au_feat
        else:
            fused = self.vproj(cat)                   # concat -> proj (default)
        listener_feat = self.norm_v(fused)

        # speaker bidir cross-attention (v3)
        a2t, _ = self.cross_attn_a2t(query=h_a, key=h_t, value=h_t)
        a2t = self.norm_a2t(self.proj_a2t(a2t))
        t2a, _ = self.cross_attn_t2a(query=h_t, key=h_a, value=h_a)
        t2a = self.norm_t2a(self.proj_t2a(t2a))
        sp_a = self.speaker_drop(a2t.mean(dim=1))
        sp_t = self.speaker_drop(t2a.mean(dim=1))
        if self.training and self.speaker_drop_p > 0 and torch.rand(1).item() < self.speaker_drop_p:
            sp_a = torch.zeros_like(sp_a)
            sp_t = torch.zeros_like(sp_t)

        # fusion (v3 Fix3)
        q  = listener_feat.unsqueeze(1)
        kv = torch.stack([sp_a, sp_t], dim=1)
        attn_out, _ = self.cross_attn_fusion(query=q, key=kv, value=kv)
        features = self.fuse_drop(self.norm_fusion(attn_out.squeeze(1) + listener_feat))

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()
        return features, emos_out, vals_out, interloss
