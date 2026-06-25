'''
Cross-Role Attention for MER-Cross.

Train-test gap: model trained on speaker emotion, tested on listener emotion.
At test time, audio+text come from the speaker while video comes from the listener.

Training strategy: Dynamic Modality Dropout — randomly zero each modality combination
so the model learns to predict without reliable audio/text.

Inference strategy: Traverse Inference — run each test sample 3 times with different
masking schemas and pick the prediction with highest confidence:
  Pass 1: [audio, text, video]  (full input)
  Pass 2: [zeros, text, video]  (suspect audio unreliable)
  Pass 3: [audio, zeros, video] (suspect text unreliable)
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

        self.traverse_inference = getattr(args, 'traverse_inference', True)
        # 'max_conf'  : pick pass with highest max-softmax (original)
        # 'soft_vote' : weighted average of softmax probs (0.6/0.2/0.2)
        # 'adaptive'  : trust full pass if conf > threshold, else fallback to no-audio
        self.traverse_mode      = getattr(args, 'traverse_mode',      'soft_vote')
        self.adaptive_threshold = getattr(args, 'adaptive_threshold', 0.65)

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
            feat_full = self._encode(audio, text, video)
            # Pass 2: audio masked (suspect audio is cross-person noise)
            feat_no_a = self._encode(torch.zeros_like(audio), text, video)
            # Pass 3: text masked (suspect text is cross-person noise)
            feat_no_t = self._encode(audio, torch.zeros_like(text), video)

            prob_full = F.softmax(self.fc_out_1(feat_full), dim=1)  # [B, C]
            prob_no_a = F.softmax(self.fc_out_1(feat_no_a), dim=1)
            prob_no_t = F.softmax(self.fc_out_1(feat_no_t), dim=1)

            if self.traverse_mode == 'soft_vote':
                # Weighted average: full pass keeps majority vote
                final_prob = 0.6 * prob_full + 0.2 * prob_no_a + 0.2 * prob_no_t
                emos_out   = torch.log(final_prob + 1e-8)  # back to log-space for CE loss

            elif self.traverse_mode == 'adaptive':
                # Trust full pass if confident; fallback to no-audio when ambiguous
                conf_full = prob_full.max(dim=1)[0]                          # [B]
                use_full  = (conf_full > self.adaptive_threshold).float()    # [B] 0/1
                # blend: use_full * prob_full + (1-use_full) * prob_no_a
                final_prob = use_full.unsqueeze(1) * prob_full + \
                             (1 - use_full).unsqueeze(1) * prob_no_a
                emos_out   = torch.log(final_prob + 1e-8)

            else:  # 'max_conf' — original behaviour
                conf_full = prob_full.max(dim=1)[0]
                conf_no_a = prob_no_a.max(dim=1)[0]
                conf_no_t = prob_no_t.max(dim=1)[0]
                confs     = torch.stack([conf_full, conf_no_a, conf_no_t], dim=1)
                best      = confs.argmax(dim=1)
                probs     = torch.stack([prob_full, prob_no_a, prob_no_t], dim=1)  # [B,3,C]
                final_prob = probs[torch.arange(probs.size(0), device=probs.device), best]
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
