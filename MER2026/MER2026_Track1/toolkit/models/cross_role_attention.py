'''
Cross-Role Attention for MER-Cross.

Train-test gap: model trained on speaker emotion, tested on listener emotion.
At test time, audio+text come from the speaker while video comes from the listener.

Two complementary strategies:
  1. Dynamic Modality Dropout — randomly zero each modality combination so the
     model learns to predict without reliable audio/text.
  2. Gradient Scaling — scale down audio/text gradients in the backward pass so
     the video branch gets proportionally more gradient signal and is optimised
     deeper, without distorting the forward-pass feature geometry.
'''
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder


class GradScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None


def scale_grad(x, alpha):
    return GradScaleFunction.apply(x, alpha)


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

        # Dynamic Modality Dropout probabilities (must sum <= 1.0)
        self.p_mask_a  = getattr(args, 'p_mask_a',  0.30)
        self.p_mask_t  = getattr(args, 'p_mask_t',  0.20)
        self.p_mask_at = getattr(args, 'p_mask_at', 0.20)

        # Gradient scaling: alpha < 1.0 slows optimisation of that branch
        self.alpha_audio = getattr(args, 'alpha_audio', 0.3)
        self.alpha_text  = getattr(args, 'alpha_text',  0.5)
        # video alpha is always 1.0 (full gradient)

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

        a_h = self.audio_encoder(audio)
        t_h = self.text_encoder(text)
        v_h = self.video_encoder(video)

        # Scale down audio/text gradients; video gets full gradient signal
        a_h = scale_grad(a_h, self.alpha_audio)
        t_h = scale_grad(t_h, self.alpha_text)

        attn_h   = self.attention_mlp(torch.cat([a_h, t_h, v_h], dim=1))
        attn_raw = self.fc_att(attn_h)
        attn_w   = torch.softmax(attn_raw, dim=1)                         # [B, 3]
        stacked  = torch.stack([a_h, t_h, v_h], dim=2)                   # [B, H, 3]
        features = torch.matmul(stacked, attn_w.unsqueeze(2)).squeeze(2)  # [B, H]

        emos_out  = self.fc_out_1(features)
        vals_out  = self.fc_out_2(features)
        interloss = torch.zeros(1, device=audio.device).squeeze()

        return features, emos_out, vals_out, interloss
