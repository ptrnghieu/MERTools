'''
Video-Guided Cross-Attention Fusion for MER-Cross.

MER-Cross train-test gap: audio+text from speaker, video from listener.
Video frames (listener) act as queries to selectively extract aligned context
from audio and text (speaker), anchoring speech dynamics to facial expression
timestamps and suppressing speech-active but face-static segments.

Retains Dynamic Modality Dropout + Video CORAL from CrossRoleAttention.
'''
import torch
import torch.nn as nn


class GRASPSequenceFusion(nn.Module):
    def __init__(self, args):
        super().__init__()

        audio_dim   = args.audio_dim
        text_dim    = args.text_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout     = args.dropout
        hidden_dim  = args.hidden_dim
        self.grad_clip    = args.grad_clip
        self.p_mask_a     = getattr(args, 'p_mask_a',     0.30)
        self.p_mask_t     = getattr(args, 'p_mask_t',     0.20)
        self.p_mask_at    = getattr(args, 'p_mask_at',    0.20)
        self.coral_lambda = getattr(args, 'coral_lambda', 0.10)
        num_heads         = getattr(args, 'num_heads',    4)

        # Set by main-release.py before training: CPU tensor (N_test, T_v, video_dim)
        self.test_video_feats = None

        # Input projections
        self.proj_a = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_t = nn.Sequential(nn.Linear(text_dim,  hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj_v = nn.Sequential(nn.Linear(video_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        # Guided cross-attention: Video(Q) x Audio(K,V)
        self.guided_attn_a = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(hidden_dim)

        # Guided cross-attention: Video(Q) x Text(K,V)
        self.guided_attn_t = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(hidden_dim)

        self.norm_v = nn.LayerNorm(hidden_dim)

        # Output heads
        self.fc_out_1 = nn.Linear(hidden_dim * 3, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim * 3, max(1, output_dim2))

    def _coral(self, source, target):
        ns, nt = source.size(0), target.size(0)
        cov_s = (source - source.mean(0)).T @ (source - source.mean(0)) / max(ns - 1, 1)
        cov_t = (target - target.mean(0)).T @ (target - target.mean(0)) / max(nt - 1, 1)
        return ((cov_s - cov_t) ** 2).mean()

    def forward(self, batch):
        audio = batch['audios']  # (B, T_a, audio_dim)
        text  = batch['texts']   # (B, T_t, text_dim)
        video = batch['videos']  # (B, T_v, video_dim)
        B = audio.shape[0]

        # Padding masks computed BEFORE dropout to avoid all-masked NaN.
        # pad_to_maxlen_pre_modality zero-pads at the beginning; zero rows = padded.
        a_pad_mask = (audio.abs().sum(-1) == 0)  # (B, T_a), True = padded
        t_pad_mask = (text.abs().sum(-1)  == 0)  # (B, T_t), True = padded

        # Dynamic Modality Dropout (training only, inference always uses all modalities)
        if self.training:
            r = torch.rand(1).item()
            if r < self.p_mask_a:
                audio = torch.zeros_like(audio)
            elif r < self.p_mask_a + self.p_mask_t:
                text  = torch.zeros_like(text)
            elif r < self.p_mask_a + self.p_mask_t + self.p_mask_at:
                audio = torch.zeros_like(audio)
                text  = torch.zeros_like(text)

        # Project each modality to hidden_dim
        a_proj = self.proj_a(audio)  # (B, T_a, H)
        t_proj = self.proj_t(text)   # (B, T_t, H)
        v_proj = self.proj_v(video)  # (B, T_v, H)

        # Safety: if all keys masked for a sample, unmask all to prevent softmax NaN.
        # This can happen when audio/text features are all-zero (e.g., empty transcription).
        a_pad_mask_safe = a_pad_mask.clone()
        a_pad_mask_safe[a_pad_mask.all(dim=1)] = False
        t_pad_mask_safe = t_pad_mask.clone()
        t_pad_mask_safe[t_pad_mask.all(dim=1)] = False

        # Video-guided cross-attention over Audio
        a_ctx, _ = self.guided_attn_a(
            query=v_proj, key=a_proj, value=a_proj,
            key_padding_mask=a_pad_mask_safe,
        )  # (B, T_v, H)

        # Video-guided cross-attention over Text
        t_ctx, _ = self.guided_attn_t(
            query=v_proj, key=t_proj, value=t_proj,
            key_padding_mask=t_pad_mask_safe,
        )  # (B, T_v, H)

        # LayerNorm then temporal mean pooling -> utterance vector
        v_out = self.norm_v(v_proj)           # (B, T_v, H)
        a_out = self.norm_a(a_ctx)            # (B, T_v, H)
        t_out = self.norm_t(t_ctx)            # (B, T_v, H)

        fused    = torch.cat([v_out, a_out, t_out], dim=-1)  # (B, T_v, 3H)
        features = fused.mean(dim=1)                          # (B, 3H)

        interloss = torch.zeros(1, device=audio.device).squeeze()

        # Video CORAL: align train vs test video encoder distributions
        if self.training and self.test_video_feats is not None and self.coral_lambda > 0:
            idx      = torch.randperm(self.test_video_feats.size(0))[:B]
            test_v   = self.test_video_feats[idx].to(audio.device)  # (B, T_v, video_dim)
            test_v_h = self.proj_v(test_v).mean(dim=1)              # (B, H)
            train_v_h = v_proj.mean(dim=1)                          # (B, H)
            interloss = interloss + self.coral_lambda * self._coral(train_v_h, test_v_h)

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        return features, emos_out, vals_out, interloss
