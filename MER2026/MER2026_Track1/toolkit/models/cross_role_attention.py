'''
Cross-Role Attention for MER-Cross.

Train-test gap: model trained on speaker emotion, tested on listener emotion.
At test time, audio+text come from the speaker while video comes from the listener.

Two complementary strategies:
  1. Dynamic Modality Dropout — randomly zero each modality combination during training
     so the model learns to predict without reliable audio/text.
  2. Video CORAL Alignment — penalise the Frobenius distance between the covariance
     matrices of train and test VIDEO encoder outputs. Video is the Anchor modality
     (stable across speaker/listener roles); aligning its distribution bridges the
     domain gap without touching the noisy audio/text Drift modalities.

Inference is always a clean full pass (all modalities, no masking).
CORAL and Dynamic Dropout are training-only regularisers.
'''
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder, LSTMEncoder


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
        self.coral_lambda = getattr(args, 'coral_lambda', 0.1)

        # Set by main-release.py before training: CPU tensor [N_test, video_dim] or [N_test, seq_len, video_dim]
        self.test_video_feats = None

        feat_type = getattr(args, 'feat_type', 'utt')
        if feat_type in ['frm_align', 'frm_unalign']:
            self.audio_encoder = LSTMEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = LSTMEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = LSTMEncoder(video_dim, hidden_dim, dropout)
        else:
            self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
            self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
            self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)
        self.fc_att        = nn.Linear(hidden_dim, 3)
        self.fc_out_1      = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2      = nn.Linear(hidden_dim, output_dim2)

    def _coral(self, source, target):
        """CORAL loss: mean squared Frobenius norm of covariance difference."""
        ns, nt = source.size(0), target.size(0)
        cov_s = (source - source.mean(0)).T @ (source - source.mean(0)) / (ns - 1)
        cov_t = (target - target.mean(0)).T @ (target - target.mean(0)) / (nt - 1)
        return ((cov_s - cov_t) ** 2).mean()

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']
        B     = audio.shape[0]

        # Dynamic Modality Dropout: training only, inference is always a clean full pass
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

        attn_h   = self.attention_mlp(torch.cat([a_h, t_h, v_h], dim=1))
        attn_raw = self.fc_att(attn_h)
        attn_w   = torch.softmax(attn_raw, dim=1)
        stacked  = torch.stack([a_h, t_h, v_h], dim=2)
        features = torch.matmul(stacked, attn_w.unsqueeze(2)).squeeze(2)

        interloss = torch.zeros(1, device=audio.device).squeeze()

        # Video CORAL: align train video encoder outputs with test video encoder outputs
        if self.training and self.test_video_feats is not None and self.coral_lambda > 0:
            idx      = torch.randperm(self.test_video_feats.size(0))[:B]
            test_v   = self.test_video_feats[idx].to(audio.device)
            test_v_h = self.video_encoder(test_v)
            interloss = interloss + self.coral_lambda * self._coral(v_h, test_v_h)

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        return features, emos_out, vals_out, interloss
