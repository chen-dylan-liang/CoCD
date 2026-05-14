from typing import Optional

from torch.optim.optimizer import ParamsT
import torch
import torch.optim as optim

from zeroth_order_optim import ZerothOrderOptimizer, finite_difference


class CoCD(ZerothOrderOptimizer, optim.Optimizer):
    """Coherent Coordinate Descent optimizer from the ICML 2026 CoCD paper.

    CoCD is an efficient deterministic zeroth-order optimizer. 
    At each iteration, it refreshes a small portion of the cyclic gradient buffer with finite differences, 
    optionally decays the rest of the buffer, and applies the buffer as the descent direction.

    The buffer turns stale partial gradients into warm starts. `momentum=1` preserves
    each estimate until it is refreshed, `momentum=0` recovers Block Cyclic
    Coordinate Descent without gradient reuse, and values between 0 and 1
    exponentially down-weight older estimates. `eps` is the finite-difference
    radius from the paper; appropriately larger radii can act as implicit
    landscape smoothing for faster convergence and better stability.

    The optimizer expects a closure that recomputes and returns a scalar
    objective at the current parameter values. `step`, `approximate_gradient`,
    and `optimize` run under `torch.no_grad()` because this optimizer uses only
    function evaluations and never calls backpropagation.
    """

    batched = False

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        init_grad: Optional[torch.Tensor] = None,
        eps: float = 1e-1,
        compute_budget: int = 1,
        memory_budget: Optional[int] = None,
        momentum: float = 1.0,
        central_difference: bool = True,
    ):
        """Initialize CoCD.

        Args:
            params: Model parameters to optimize.
            lr: Learning rate used when applying the gradient buffer.
            weight_decay: Decoupled weight decay applied during parameter
                updates. Negative values are clamped to zero.
            init_grad: Optional initial flattened gradient buffer. Its number
                of elements must match `memory_budget`. Defaults to a zero buffer.
            eps: Finite-difference perturbation radius, also interpreted as
                the smoothing radius in the paper.
            compute_budget: Number of scalar coordinates refreshed per
                optimizer step.
            memory_budget: Number of gradient estimates kept in the FIFO
                buffer. Defaults to store one estimate per scalar parameter.
            momentum: Decay applied to the buffer before new coordinates are
                refreshed. `1.0` means full reuse; `0.0` recovers Block Cyclic
                Coordinate Descent.
            central_difference: If true, use symmetric central finite
                differences; otherwise use a one-sided finite difference.
        """
        self.params = list(params)
        if not self.params:
            raise ValueError("CoCD requires at least one parameter tensor.")

        if compute_budget <= 0:
            raise ValueError("compute_budget must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.device = self.params[0].device
        self.lr = lr
        self.weight_decay = max(weight_decay, 0.0)
        self.eps = eps
        self.compute_budget = int(compute_budget)
        self.momentum = momentum
        self.central_difference = central_difference

        self.parameter_sizes = [param.numel() for param in self.params]
        self.cumulative_parameter_sizes = torch.zeros(len(self.params) + 1, dtype=torch.long)
        for index, size in enumerate(self.parameter_sizes):
            self.cumulative_parameter_sizes[index + 1] = (
                self.cumulative_parameter_sizes[index] + size
            )
        self.num_parameters = int(self.cumulative_parameter_sizes[-1].item())
        if self.num_parameters == 0:
            raise ValueError("CoCD requires at least one scalar parameter.")

        self.memory_budget = self.num_parameters if memory_budget is None else int(memory_budget)
        if self.memory_budget <= 0:
            raise ValueError("memory_budget must be positive.")
        if self.memory_budget > self.num_parameters:
            raise ValueError("memory_budget cannot exceed the number of scalar parameters.")

        if init_grad is None:
            self.grad = torch.zeros(self.memory_budget, device=self.device)
        else:
            if init_grad.numel() != self.memory_budget:
                raise ValueError("init_grad must have memory_budget elements.")
            self.grad = init_grad.reshape(-1).to(self.device)

        # Pointers for refreshing gradient estimates.
        self.current_parameter_index = 0
        self.current_weight_index = 0
        self.current_gradient_index = 0

        # Pointers for mapping the rolling gradient buffer back to parameters.
        self.memory_is_full = False
        self.gradient_offset = 0
        self.parameter_offset = 0
        self.weight_offset = 0

        defaults = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "eps": self.eps,
        }
        optim.Optimizer.__init__(self, self.params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("CoCD requires a closure for zeroth-order updates.")

        self.approximate_gradient(closure)
        self.optimize()
        return None

    @torch.no_grad()
    def approximate_gradient(self, closure):
        self.grad.mul_(self.momentum)
        for _ in range(self.compute_budget):
            self.grad[self.current_gradient_index] = finite_difference(
                self.params[self.current_parameter_index],
                self.current_weight_index,
                closure,
                self.eps,
                self.central_difference,
            )
            self.advance_pointers()

    def advance_pointers(self):
        if self.memory_is_full:
            self.gradient_offset = (self.gradient_offset + 1) % self.memory_budget
            self.weight_offset += 1
            if self.weight_offset == self.params[self.parameter_offset].numel():
                self.weight_offset = 0
                self.parameter_offset = (self.parameter_offset + 1) % len(self.params)

        if not self.memory_is_full and self.current_gradient_index + 1 == self.memory_budget:
            self.memory_is_full = True

        self.current_gradient_index = (
            self.current_gradient_index + 1
        ) % self.memory_budget
        self.current_weight_index += 1
        if self.current_weight_index == self.params[self.current_parameter_index].numel():
            self.current_weight_index = 0
            self.current_parameter_index = (
                self.current_parameter_index + 1
            ) % len(self.params)

    @torch.no_grad()
    def optimize(self):
        gradient_offset = self.gradient_offset
        parameter_offset = self.parameter_offset
        weight_offset = self.weight_offset
        updated_weights = 0

        while updated_weights < self.memory_budget:
            weights_available = self.params[parameter_offset].numel() - weight_offset
            weights_to_update = min(
                weights_available,
                self.memory_budget - updated_weights,
            )
            gradient_indices = (
                torch.arange(weights_to_update, device=self.device) + gradient_offset
            ) % self.memory_budget
            weights = self.params[parameter_offset].view(-1)[
                weight_offset:weight_offset + weights_to_update
            ]
            if self.weight_decay > 0:
                weights.mul_(1 - self.lr * self.weight_decay)
            weights.sub_(self.lr * self.grad[gradient_indices].to(dtype=weights.dtype))

            gradient_offset = (gradient_offset + weights_to_update) % self.memory_budget
            parameter_offset = (parameter_offset + 1) % len(self.params)
            weight_offset = 0
            updated_weights += weights_to_update

    def reset(self):
        self.grad.zero_()
        self.current_parameter_index = 0
        self.current_weight_index = 0
        self.current_gradient_index = 0
        self.memory_is_full = False
        self.gradient_offset = 0
        self.parameter_offset = 0
        self.weight_offset = 0

    def state_dict(self):
        state = optim.Optimizer.state_dict(self)
        state["cocd_state"] = {
            "grad": self.grad,
            "current_parameter_index": self.current_parameter_index,
            "current_weight_index": self.current_weight_index,
            "current_gradient_index": self.current_gradient_index,
            "memory_is_full": self.memory_is_full,
            "gradient_offset": self.gradient_offset,
            "parameter_offset": self.parameter_offset,
            "weight_offset": self.weight_offset,
        }
        return state

    def load_state_dict(self, state_dict):
        if state_dict is None:
            return

        state_dict = state_dict.copy()
        cocd_state = state_dict.pop("cocd_state", {})
        optim.Optimizer.load_state_dict(self, state_dict)

        grad = cocd_state.get("grad")
        if grad is not None:
            self.grad = grad.reshape(-1).to(self.device)
        self.current_parameter_index = cocd_state.get(
            "current_parameter_index",
            self.current_parameter_index,
        )
        self.current_weight_index = cocd_state.get(
            "current_weight_index",
            self.current_weight_index,
        )
        self.current_gradient_index = cocd_state.get(
            "current_gradient_index",
            self.current_gradient_index,
        )
        self.memory_is_full = cocd_state.get("memory_is_full", self.memory_is_full)
        self.gradient_offset = cocd_state.get("gradient_offset", self.gradient_offset)
        self.parameter_offset = cocd_state.get("parameter_offset", self.parameter_offset)
        self.weight_offset = cocd_state.get("weight_offset", self.weight_offset)
