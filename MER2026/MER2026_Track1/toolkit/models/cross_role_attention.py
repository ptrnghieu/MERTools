'''
Cross-Role Attention for MER-Cross.

Train-test gap: model trained on speaker emotion, tested on listener emotion.
At test time, audio+text come from the speaker while video comes from the listener.

Strategy: Dynamic Modality Dropout — independently mask each modality combination
with probabilities derived from transferability (HuBERT overfits most → masked most;
video is most reliable → never masked).

Default schedule:
  30% — mask audio only  (force model to use text + video)
  20% — mask text only   (force model to use audio + video)
  20% — mask audio+text  (force model to use video only — core cross-role condition)
  30% — keep all three
'''
import torch
import torch.nn as nn
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

        # p_mask_a + p_mask_t + p_mask_at <= 1.0; remainder = keep all
        self.p_mask_a  = getattr(args, 'p_mask_a',  0.30)
        self.p_mask_t  = getattr(args, 'p_mask_t',  0.20)
        self.p_mask_at = getattr(args, 'p_mask_at', 0.20)

        self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
        self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
        self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)
        self.fc_att        = nn.Linear(hidden_dim, 3)
        self.fc_out_1      = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2      = nn.Linear(hidden_dim, output_dim2)

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
            # else: keep all three (remaining probability)

        a_h = self.audio_encoder(audio)
        t_h = self.text_encoder(text)
        v_h = self.video_encoder(video)

        attn_h   = self.attention_mlp(torch.cat([a_h, t_h, v_h], dim=1))
        attn_raw = self.fc_att(attn_h)
        attn_w   = torch.softmax(attn_raw, dim=1)                         # [B, 3]
        stacked  = torch.stack([a_h, t_h, v_h], dim=2)                   # [B, H, 3]
        features = torch.matmul(stacked, attn_w.unsqueeze(2)).squeeze(2)  # [B, H]

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()

        return features, emos_out, vals_out, interloss
