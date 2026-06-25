import torch
import torch.nn as nn
import torch.nn.functional as F

# classification loss
class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1) # [n_samples, n_classes]
        target = target.long()        # [n_samples]
        loss = self.loss(pred, target) / len(pred)
        return loss

# supervised contrastive loss (Khosla et al. 2020)
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [B, D], labels: [B]
        device = features.device
        features = F.normalize(features, dim=1)
        B = features.shape[0]

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B] same-class mask

        # logits: [B, B], remove self-similarity on diagonal
        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # numerical stability

        # exclude diagonal (self-contrast)
        self_mask = torch.eye(B, dtype=torch.bool, device=device)
        mask[self_mask] = 0
        logits[self_mask] = float('-inf')

        # log-softmax over all negatives + positives
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        # mean over positives per anchor; skip anchors with no positives
        n_positives = mask.sum(dim=1)
        valid = n_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (n_positives + 1e-9)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


# regression loss
class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss
