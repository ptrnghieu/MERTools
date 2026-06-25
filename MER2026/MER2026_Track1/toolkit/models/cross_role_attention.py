'''
Cross-Role Attention for MER-Cross.

Train-test gap: model trained on speaker emotion, tested on listener emotion.
At test time, audio+text come from the speaker while video comes from the listener.

Strategy: modality dropout — randomly zero out audio, text, or both during training.
Forces the model to predict from video alone when other modalities are absent,
directly simulating the unreliable-AT condition at test time.

dropout_mode controls what gets zeroed each trigger:
  'at'   — zero both audio AND text (closest to test condition)
  'a'    — zero audio only
  't'    — zero text only
  'any'  — randomly pick one of the three above each trigger
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

        self.modal_shuffle_p   = getattr(args, 'modal_shuffle_p',   0.5)
        self.modal_dropout_mode = getattr(args, 'modal_dropout_mode', 'at')

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
        B     = audio.shape[0]

        if self.training and torch.rand(1).item() < self.modal_shuffle_p:
            mode = self.modal_dropout_mode
            if mode == 'any':
                mode = ['at', 'a', 't'][torch.randint(3, (1,)).item()]
            if mode in ('at', 'a'):
                audio = torch.zeros_like(audio)
            if mode in ('at', 't'):
                text  = torch.zeros_like(text)

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
