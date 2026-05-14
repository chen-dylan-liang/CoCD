from abc import ABC, abstractmethod
import torch

class ZerothOrderOptimizer(ABC):

    @abstractmethod
    def approximate_gradient(self, closure):
        pass

    @abstractmethod
    def optimize(self):
        pass
    
    @torch.no_grad()
    def step(self, closure):
        self.approximate_gradient(closure)
        self.optimize()
        return None

    @abstractmethod
    def state_dict(self):
        pass
    
    @property
    @abstractmethod
    def batched(self):
        pass
    
    @abstractmethod
    def reset():
        pass   


def finite_difference(param, idx, closure, eps=1e-6, central=True):
    w = param.view(-1)[idx]
    # eps = eps * torch.abs(w) if torch.abs(w) > 1e-6 else eps
    if central:
        w += eps
        y1 = closure()
        w -= 2*eps
        y0 = closure()
        w += eps
        return (y1 - y0)/(2*eps)
    else:
        w += eps
        y1 = closure()
        w -= eps
        y0 = closure()
        return (y1 - y0) / eps





