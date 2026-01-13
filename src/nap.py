import torch
import torch.nn as nn


class WeightProjector:
    """Project Conv/Linear weights back to their initial L2 radius."""

    def __init__(self, model: nn.Module, eps: float = 1e-12) -> None:
        self.eps = eps
        self.radii: dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.radii[name] = module.weight.data.norm(2).detach().clone()

    @torch.no_grad()
    def project_(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in self.radii:
                weight = module.weight.data
                norm = weight.norm(2)
                target = self.radii[name].to(weight.device)
                module.weight.data = weight * (target / (norm + self.eps))
