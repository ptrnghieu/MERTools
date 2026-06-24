'''
Two-stage model for MER-Cross, inspired by CSCER (Fatima & Erzin, Interspeech 2017).

Key insight from CSCER: in dyadic interactions, the emotional state of one person
(speaker) provides useful context for predicting the other person's emotion (listener).

Stage 1: Audio + Text of speaker → Speaker emotion distribution
Stage 2: Speaker emotion distribution + Listener video → Listener emotion

At test time, audio+text come from the speaker (s1). Stage 1 predicts s1's emotion.
Stage 2 uses s1's predicted emotion as "emotional context" + s2's video to predict
s2's reaction.

Cross-role simulation: randomly shuffle Stage 1 outputs across samples during
training, so Stage 2 learns to use video as the primary cue and speaker emotion
only as soft context (not a reliable shortcut).
'''
import torch
import torch.nn as nn
from .modules.encoder import MLPEncoder


class TwoStageModel(nn.Module):
    def __init__(self, args):
        super(TwoStageModel, self).__init__()

        text_dim    = args.text_dim
        audio_dim   = args.audio_dim
        video_dim   = args.video_dim
        output_dim1 = args.output_dim1
        output_dim2 = args.output_dim2
        dropout     = args.dropout
        hidden_dim  = args.hidden_dim
        self.grad_clip = args.grad_clip

        self.modal_shuffle_p = getattr(args, 'modal_shuffle_p', 0.5)

        # Stage 1: audio + text → speaker emotion
        self.audio_encoder  = MLPEncoder(audio_dim, hidden_dim, dropout)
        self.text_encoder   = MLPEncoder(text_dim,  hidden_dim, dropout)
        self.stage1_mlp     = MLPEncoder(hidden_dim * 2, hidden_dim, dropout)
        self.stage1_head    = nn.Linear(hidden_dim, output_dim1)

        # Stage 2: video + speaker emotion probs → listener emotion
        self.video_encoder  = MLPEncoder(video_dim, hidden_dim, dropout)
        self.stage2_mlp     = MLPEncoder(hidden_dim + output_dim1, hidden_dim, dropout)
        self.fc_out_1       = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2       = nn.Linear(hidden_dim, output_dim2)

    def forward(self, batch):
        audio_feat = batch['audios']
        text_feat  = batch['texts']
        video_feat = batch['videos']

        # ---- Stage 1: speaker emotion from audio + text ----
        audio_hidden = self.audio_encoder(audio_feat)
        text_hidden  = self.text_encoder(text_feat)
        at_hidden    = self.stage1_mlp(torch.cat([audio_hidden, text_hidden], dim=1))
        speaker_logits = self.stage1_head(at_hidden)          # [B, 6]
        speaker_probs  = torch.softmax(speaker_logits, dim=1) # [B, 6]

        # Cross-role simulation: shuffle speaker_probs across samples.
        # Forces Stage 2 to learn from video alone when speaker context is "wrong",
        # mirroring test-time conditions where audio/text come from a different person.
        if self.training:
            B = speaker_probs.shape[0]
            if B > 1 and torch.rand(1).item() < self.modal_shuffle_p:
                perm = torch.randperm(B, device=speaker_probs.device)
                ctx = speaker_probs[perm].detach()
            else:
                ctx = speaker_probs
        else:
            ctx = speaker_probs  # at test time: use s1's actual emotion prediction

        # ---- Stage 2: listener emotion from video + speaker emotion context ----
        video_hidden  = self.video_encoder(video_feat)
        stage2_hidden = self.stage2_mlp(torch.cat([video_hidden, ctx], dim=1))

        emos_out  = self.fc_out_1(stage2_hidden)
        vals_out  = self.fc_out_2(stage2_hidden)
        interloss = torch.tensor(0).cuda()

        return stage2_hidden, emos_out, vals_out, interloss
