'''
Cross-Role Transferability Attention for MER-Cross.

Problem: train on speaker emotion (audio+text+video of s1, label=s1 emotion),
test on listener emotion (audio+text of s1 + video of s2, predict s2 emotion).
Audio/text come from the WRONG person at test time → ~26pt train-test gap.

Solution: during training, randomly shuffle audio/text across samples to simulate
the cross-role mismatch. Model learns to rely on video (role-invariant facial
expression) rather than audio/text (role-specific speech features).
'''
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder
from toolkit.utils.loss import SupConLoss


class CrossRoleAttention(nn.Module):
    def __init__(self, args):
        super(CrossRoleAttention, self).__init__()

        text_dim    = args.text_dim
        audio_dim   = args.audio_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout     = args.dropout
        hidden_dim  = args.hidden_dim
        self.grad_clip = args.grad_clip

        self.modal_shuffle_p = getattr(args, 'modal_shuffle_p', 0.5)
        self.transfer_lambda = getattr(args, 'transfer_lambda', 0.0)
        self.supcon_lambda   = getattr(args, 'supcon_lambda', 0.1)
        self.supcon_loss_fn  = SupConLoss(temperature=0.07)

        self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
        self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
        self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * 3, hidden_dim, dropout)
        self.fc_att   = nn.Linear(hidden_dim, 3)
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio_feat = batch['audios']
        text_feat  = batch['texts']
        video_feat = batch['videos']

        # Cross-role simulation: shuffle audio/text across samples in batch.
        # This mimics test-time conditions where audio+text belong to the speaker
        # while video (and the label) belongs to the listener.
        if self.training:
            B = audio_feat.shape[0]
            if B > 1 and torch.rand(1).item() < self.modal_shuffle_p:
                perm = torch.randperm(B, device=audio_feat.device)
                audio_feat = audio_feat[perm]
            if B > 1 and torch.rand(1).item() < self.modal_shuffle_p:
                perm = torch.randperm(B, device=text_feat.device)
                text_feat = text_feat[perm]

        audio_hidden = self.audio_encoder(audio_feat)
        text_hidden  = self.text_encoder(text_feat)
        video_hidden = self.video_encoder(video_feat)

        multi_hidden1 = torch.cat([audio_hidden, text_hidden, video_hidden], dim=1)
        attention_h   = self.attention_mlp(multi_hidden1)
        attention_raw = self.fc_att(attention_h)

        # attention_weights: [B, 3] — indices 0=audio, 1=text, 2=video
        attention_weights = torch.softmax(attention_raw, dim=1)
        attention = torch.unsqueeze(attention_weights, 2)  # [B, 3, 1]

        multi_hidden2 = torch.stack([audio_hidden, text_hidden, video_hidden], dim=2)
        fused_feat = torch.matmul(multi_hidden2, attention)  # [B, H, 1]
        features   = fused_feat.squeeze(axis=2)              # [B, H]

        emos_out = self.fc_out_1(features)
        vals_out = self.fc_out_2(features)

        interloss = torch.tensor(0.0).cuda()

        if self.training:
            # Transferability regularization: penalise attention on audio+text
            if self.transfer_lambda > 0:
                interloss = interloss + self.transfer_lambda * attention_weights[:, :2].sum(dim=1).mean()

            # Supervised contrastive loss on video_hidden only.
            # video is never shuffled → always aligned with the label.
            # This pushes video encoder to learn emotion-discriminative clusters
            # without interference from shuffled audio/text.
            if self.supcon_lambda > 0 and 'emos' in batch:
                interloss = interloss + self.supcon_lambda * self.supcon_loss_fn(video_hidden, batch['emos'])

        return features, emos_out, vals_out, interloss
