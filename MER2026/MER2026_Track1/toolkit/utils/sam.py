"""
Sharpness-Aware Minimization (Foret et al. 2021) + ASAM (Kwon et al. 2021).

SAM seeks parameters that lie in flat loss regions: it perturbs the weights
toward the local worst case (w + e(w)) before taking the base-optimizer step,
so the update minimizes the loss over a neighborhood rather than a single sharp
point. Flat minima generalize better -- valuable for small datasets with a
train/test distribution shift (MER-Cross role shift, ~9k train samples).

Usage (two forward-backward passes per step):
    optimizer = SAM(model.parameters(), torch.optim.AdamW,
                    rho=0.05, adaptive=False, lr=1e-4, weight_decay=1e-5)
    # step 1: gradient at w
    loss_fn().backward()
    optimizer.first_step(zero_grad=True)      # ascend to w + e(w)
    # step 2: gradient at w + e(w)
    loss_fn().backward()
    optimizer.second_step(zero_grad=True)     # restore w, base_optimizer.step()

adaptive=True -> ASAM (scale-invariant perturbation), usually a larger rho
(e.g. 0.5-2.0) than plain SAM (0.05-0.1).

Reference implementation: davda54/sam.
"""
import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0, f'Invalid rho, should be non-negative: {rho}'
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['old_p'] = p.data.clone()
                e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)                      # climb to local maximum "w + e(w)"
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']  # restore "w" from "w + e(w)"
        self.base_optimizer.step()               # sharpness-aware update
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
