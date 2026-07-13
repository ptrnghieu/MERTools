import torch
import torch.nn as nn
import torch.nn.functional as F

# classification loss
class CELoss(nn.Module):

    def __init__(self, weight=None):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum', weight=weight)

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1) # [n_samples, n_classes]
        target = target.long()        # [n_samples]
        loss = self.loss(pred, target) / len(pred)
        return loss


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017):
        FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy, well-classified examples so training focuses on hard /
    rare cases -- helps the under-predicted minority classes (e.g. surprise,
    worried) in MER-Cross. gamma=0 reduces to (weighted) cross-entropy.
    weight: optional per-class alpha tensor (idx order), e.g. inverse-frequency.
    Normalized as sum/len(pred) to match CELoss scale.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight              # per-class alpha tensor or None

    def forward(self, pred, target):
        target = target.long()                                   # (N,)
        logp = F.log_softmax(pred, dim=1)                        # (N, C)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)   # (N,)
        pt = logpt.exp()
        focal = (1 - pt) ** self.gamma * (-logpt)                # (N,)
        if self.weight is not None:
            focal = focal * self.weight.gather(0, target)
        return focal.sum() / len(pred)


class LabelSmoothingCELoss(nn.Module):

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_prob = F.log_softmax(pred, dim=1)           # (N, C)
        target = target.long()
        nll = F.nll_loss(log_prob, target, reduction='sum')
        smooth = -log_prob.sum(dim=1).sum()             # uniform over all classes
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth / n_classes
        return loss / len(pred)

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
