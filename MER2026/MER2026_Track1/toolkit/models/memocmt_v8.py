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


class MemoCMTV8(nn.Module):
    """
    Reliability-Guided Affective Complementation (adapted from VISAFF,
    arXiv:2605.18547). Video (listener) is the anchor; speaker audio/text are
    visual-guided retrieved references injected as a residual gated by the
    video's own reliability.

    Unlike v3 (symmetric speaker self-mixing then fusion), v8:
      - drops the a2t/t2a speaker branch;
      - retrieves audio/text references with the visual anchor as query;
      - forms a residual complement Delta from (ref - visual);
      - gates the complement by (1 - c_v), where c_v is the max-softmax
        confidence of an auxiliary video-only classifier (detached);
      - when the listener video is confident, speaker cues are suppressed;
        when it is uncertain, contagion cues from audio/text complement it.

    Runs identically at train and eval (the gate is part of inference); the
    only train-only term is the auxiliary video-only CE, returned via interloss.
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
        self.feat_type   = getattr(args, 'feat_type',   'utt')
        self.lambda_aux  = getattr(args, 'lambda_aux',  0.3)

        num_heads = max(1, hidden_dim // 64)

        if self.feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMSeqEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMSeqEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMSeqEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        # visual anchor pooling
        self.listener_pool = LearnableQueryPooling(hidden_dim, num_heads, dropout)

        # visual-guided retrieval of speaker references (Q = visual anchor)
        self.retr_a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.retr_t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_t = nn.LayerNorm(hidden_dim)

        # residual complement from (ref - visual)
        self.delta_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm_star = nn.LayerNorm(hidden_dim)

        # auxiliary video-only classifier -> reliability score c_v
        self.aux_head = nn.Linear(hidden_dim, output_dim1)

        # final classifier over [h_v* ; t_ref ; a_ref]
        self.fuse_drop = nn.Dropout(dropout)
        self.fc_out_1 = nn.Linear(3 * hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(3 * hidden_dim, output_dim2)

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

        # visual anchor
        h_v_vec = self.listener_pool(h_v)                 # (B, H)
        q = h_v_vec.unsqueeze(1)                          # (B, 1, H)

        # visual-guided retrieval of speaker references
        a_ref, _ = self.retr_a(query=q, key=h_a, value=h_a)
        a_ref = self.norm_a(a_ref.squeeze(1))             # (B, H)
        t_ref, _ = self.retr_t(query=q, key=h_t, value=h_t)
        t_ref = self.norm_t(t_ref.squeeze(1))             # (B, H)

        # residual complement
        delta = self.delta_mlp(torch.cat([t_ref - h_v_vec, a_ref - h_v_vec], dim=-1))  # (B, H)

        # visual reliability from auxiliary video-only head
        aux_logits = self.aux_head(h_v_vec)               # (B, C)
        c_v = torch.softmax(aux_logits, dim=1).max(dim=1, keepdim=True).values.detach()  # (B,1)

        # reliability-gated complemented visual
        h_v_star = self.norm_star(h_v_vec + (1.0 - c_v) * delta)   # (B, H)

        # final representation + classifier
        features = self.fuse_drop(torch.cat([h_v_star, t_ref, a_ref], dim=-1))  # (B, 3H)
        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        interloss = torch.zeros(1, device=audio.device).squeeze()
        if self.training and 'emos' in batch:
            interloss = self.lambda_aux * F.cross_entropy(aux_logits, batch['emos'])
        return features, emos_out, vals_out, interloss
