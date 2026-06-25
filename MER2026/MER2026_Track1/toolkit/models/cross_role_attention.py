'''
Cross-Role Attention for MER-Cross.

Train-test gap: model trained on speaker emotion, tested on listener emotion.
At test time, audio+text come from the speaker while video comes from the listener.

Training strategy: Dynamic Modality Dropout — randomly zero each modality combination
so the model learns to predict without reliable audio/text.

Inference strategy (traverse_inference=True, only at test time):
  Adaptive Video Anchor — run 2 passes and fall back to video-only when full pass is
  ambiguous (high entropy signals audio/text are causing cross-role confusion):
    Pass 1 (Full):       [audio, text, video]
    Pass 2 (Anchor):     [zeros, zeros, video]
  If entropy(full) < threshold  → trust full pass
  Else                           → use video-only anchor
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.encoder import MLPEncoder


class CrossRoleAttention(nn.Module):
    def __init__(self, args):
        super(CrossRoleAttention, self).__init__()

        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout     = args.dropout
        hidden_dim  = args.hidden_dim
        self.grad_clip = args.grad_clip

        self.p_mask_a  = getattr(args, 'p_mask_a',  0.30)
        self.p_mask_t  = getattr(args, 'p_mask_t',  0.20)
        self.p_mask_at = getattr(args, 'p_mask_at', 0.20)

        # Disabled by default; main-release.py enables only during test eval
        self.traverse_inference  = False
        self.entropy_threshold   = getattr(args, 'entropy_threshold', 1.0)

        self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
        self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
        self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)
        self.fc_att        = nn.Linear(hidden_dim, 3)
        self.fc_out_1      = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2      = nn.Linear(hidden_dim, output_dim2)

    def _encode(self, audio, text, video):
        a_h = self.audio_encoder(audio)
        t_h = self.text_encoder(text)
        v_h = self.video_encoder(video)
        attn_h   = self.attention_mlp(torch.cat([a_h, t_h, v_h], dim=1))
        attn_raw = self.fc_att(attn_h)
        attn_w   = torch.softmax(attn_raw, dim=1)
        stacked  = torch.stack([a_h, t_h, v_h], dim=2)
        features = torch.matmul(stacked, attn_w.unsqueeze(2)).squeeze(2)
        return features

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']

        if self.training:
            r = torch.rand(1).item()
            if r < self.p_mask_a:
                audio = torch.zeros_like(audio)
            elif r < self.p_mask_a + self.p_mask_t:
                text  = torch.zeros_like(text)
            elif r < self.p_mask_a + self.p_mask_t + self.p_mask_at:
                audio = torch.zeros_like(audio)
                text  = torch.zeros_like(text)
            features = self._encode(audio, text, video)

        elif self.traverse_inference:
            # Pass 1: full input
            feat_full   = self._encode(audio, text, video)
            logits_full = self.fc_out_1(feat_full)
            prob_full   = F.softmax(logits_full, dim=1)                   # [B, C]

            # Entropy per sample: high entropy = model is confused = likely cross-role noise
            entropy = -(prob_full * (prob_full + 1e-9).log()).sum(dim=1)  # [B]

            # Pass 2: video-only anchor (zeros audio AND text)
            feat_anch   = self._encode(torch.zeros_like(audio), torch.zeros_like(text), video)
            logits_anch = self.fc_out_1(feat_anch)

            # Adaptive fallback: per-sample selection
            use_full = (entropy < self.entropy_threshold).float().unsqueeze(1)  # [B, 1]
            # Blend in log-space (for compatibility with CE loss downstream)
            prob_anch  = F.softmax(logits_anch, dim=1)
            final_prob = use_full * prob_full + (1 - use_full) * prob_anch
            emos_out   = torch.log(final_prob + 1e-8)

            vals_out  = self.fc_out_2(feat_full)
            interloss = torch.zeros(1, device=audio.device).squeeze()
            return feat_full, emos_out, vals_out, interloss

        else:
            features = self._encode(audio, text, video)

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()

        return features, emos_out, vals_out, interloss
