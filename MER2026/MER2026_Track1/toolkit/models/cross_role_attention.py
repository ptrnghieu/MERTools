'''
Cross-Role Discriminative Attention for MER-Cross.

Train-test gap (~26pt): model trained on speaker emotion, tested on listener emotion.
At test time, audio+text come from the speaker while video comes from the listener.

Strategy:
  1. Cross-person pairs: create (audio_j, text_j, video_i) with label = emotion_i.
     Forces model to predict from video when audio/text are from a different person.
  2. Discriminator head: learns to detect real (same-person) vs cross-person pairs.
     At inference its score boosts video attention automatically.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.encoder import MLPEncoder
from toolkit.utils.loss import SupConLoss, CELoss


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

        self.disc_lambda       = getattr(args, 'disc_lambda',       0.1)
        self.cross_emo_lambda  = getattr(args, 'cross_emo_lambda',  1.0)
        self.video_boost_scale = getattr(args, 'video_boost_scale', 2.0)
        self.transfer_lambda   = getattr(args, 'transfer_lambda',   0.0)
        self.supcon_lambda     = getattr(args, 'supcon_lambda',     0.0)

        if self.supcon_lambda > 0:
            self.supcon_loss_fn = SupConLoss(temperature=0.07)
        self.ce_loss = CELoss()

        self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
        self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
        self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)
        self.fc_att        = nn.Linear(hidden_dim, 3)
        self.discriminator = nn.Linear(hidden_dim, 1)  # real=0, cross-person=1
        self.fc_out_1      = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2      = nn.Linear(hidden_dim, output_dim2)

    def _fuse(self, audio, text, video, disc_boost=False):
        '''
        Encode three modalities and compute attended fusion.

        disc_boost: if True, use the discriminator output to upweight video
                    before the attention softmax (used at inference time).
        Returns: (fused [B,H], attn_h [B,H], attn_w [B,3], video_h [B,H])
        '''
        a_h = self.audio_encoder(audio)
        t_h = self.text_encoder(text)
        v_h = self.video_encoder(video)

        attn_h   = self.attention_mlp(torch.cat([a_h, t_h, v_h], dim=1))
        attn_raw = self.fc_att(attn_h)  # [B, 3]: audio / text / video logits

        if disc_boost:
            # High fake_prob → model likely sees cross-person input → boost video.
            fake_prob   = torch.sigmoid(self.discriminator(attn_h))  # [B, 1]
            video_extra = fake_prob * self.video_boost_scale          # [B, 1]
            boost = torch.cat([
                torch.zeros_like(video_extra),
                torch.zeros_like(video_extra),
                video_extra,
            ], dim=1)  # [B, 3]
            attn_raw = attn_raw + boost

        attn_w  = torch.softmax(attn_raw, dim=1)                         # [B, 3]
        stacked = torch.stack([a_h, t_h, v_h], dim=2)                   # [B, H, 3]
        fused   = torch.matmul(stacked, attn_w.unsqueeze(2)).squeeze(2)  # [B, H]

        return fused, attn_h, attn_w, v_h

    def forward(self, batch):
        audio = batch['audios']
        text  = batch['texts']
        video = batch['videos']
        B     = audio.shape[0]

        interloss = torch.zeros(1, device=audio.device).squeeze()

        if self.training:
            emos = batch['emos']

            # Real pairs (original, same-person)
            real_feat, real_attn_h, real_attn_w, real_v_h = \
                self._fuse(audio, text, video)

            if B > 1:
                # Cross-person pairs: shuffle audio+text together, keep video + label.
                # label = emotion_i (video person) — same as real pair.
                # With mismatched audio/text, only video carries the correct signal.
                perm = torch.randperm(B, device=audio.device)
                fake_feat, fake_attn_h, _, _ = \
                    self._fuse(audio[perm], text[perm], video)

                # 1. Discriminator loss: real=0, cross-person=1
                if self.disc_lambda > 0:
                    real_logits = self.discriminator(real_attn_h).squeeze(1)
                    fake_logits = self.discriminator(fake_attn_h).squeeze(1)
                    disc_loss = F.binary_cross_entropy_with_logits(
                        torch.cat([real_logits, fake_logits]),
                        torch.cat([
                            torch.zeros(B, device=audio.device),
                            torch.ones(B,  device=audio.device),
                        ])
                    )
                    interloss = interloss + self.disc_lambda * disc_loss

                # 2. Emotion loss on cross-person pairs (video person's label)
                if self.cross_emo_lambda > 0:
                    cross_emos_out = self.fc_out_1(fake_feat)
                    interloss = interloss + \
                        self.cross_emo_lambda * self.ce_loss(cross_emos_out, emos)

            # 3. Penalise real-pair audio+text attention (encourage video reliance)
            if self.transfer_lambda > 0:
                interloss = interloss + \
                    self.transfer_lambda * real_attn_w[:, :2].sum(1).mean()

            # 4. SupCon on video (video is never permuted → always label-aligned)
            if self.supcon_lambda > 0:
                interloss = interloss + \
                    self.supcon_lambda * self.supcon_loss_fn(real_v_h, emos)

            features = real_feat

        else:
            # Inference: discriminator score boosts video attention weight.
            # Test pairs are genuinely cross-person → discriminator fires → video boosted.
            features, _, _, _ = self._fuse(audio, text, video, disc_boost=True)

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        return features, emos_out, vals_out, interloss
