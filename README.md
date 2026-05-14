# Coherent Coordinate Descent

This repository contains a minimal PyTorch implementation of **Coherent Coordinate Descent (CoCD)**, the zeroth-order optimizer from:

> Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization. ICML 2026.

CoCD is designed for settings where first-order gradients are unavailable or expensive to obtain, such as black-box objectives and memory-constrained on-device training.

## What CoCD Does

CoCD is a deterministic zeroth-order optimizer. Instead of sampling random perturbation directions, it cycles through parameter coordinates and maintains a FIFO cyclic buffer of coordinate-wise finite-difference estimates.

At each optimizer step, CoCD:

1. Decays the existing gradient buffer by `momentum`.
2. Refreshes `compute_budget` scalar coordinates with finite differences.
3. Applies the current gradient buffer multiplied by `learning rate` to the model parameters.

This turns stale coordinate gradients into useful warm starts. It can be seen as Block Cyclic Coordinate Descent with stale-gradient warm starts. The finite-difference radius `eps` also acts as a smoothing radius: appropriately larger values can reduce high-frequency landscape noise and improve training speed and stability.

## Files

- `cocd.py`: the `CoCD` optimizer.
- `zeroth_order_optim.py`: the base zeroth-order optimizer interface and finite-difference helper.
- `LICENSE`: MIT license.

## Installation

Install PyTorch for your platform, then use the files directly:

```bash
pip install torch
```

For a local project, keep `cocd.py` and `zeroth_order_optim.py` in the same directory, or place them on your Python path.

## Quick Start

CoCD uses a PyTorch-style optimizer interface, but the `closure` is required because the optimizer needs to re-evaluate the objective after finite-difference perturbations.

```python
import torch

from cocd import CoCD


torch.manual_seed(0)

x = torch.linspace(-1, 1, 64).unsqueeze(1)
y = 3.0 * x - 0.5

model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()

optimizer = CoCD(
    model.parameters(),
    lr=1e-2,
    eps=1e-2,
    compute_budget=8,
    momentum=0.99,
)

for step in range(200):
    def closure():
        return loss_fn(model(x), y)

    loss = closure()
    optimizer.step(closure)

    if step % 50 == 0:
        print(f"step={step:03d} loss={loss.item():.6f}")
```

No `loss.backward()` call is needed. CoCD runs under `torch.no_grad()` and uses function evaluations only.

## Closure Requirements

The closure must:

- Return a scalar PyTorch tensor.
- Recompute the objective using the current parameter values.
- Avoid calling `backward()`.
- Use the same mini-batch for all finite-difference evaluations inside one optimizer step.

For stochastic models, keep the objective as deterministic as possible within a step. Random augmentations, dropout, or BatchNorm running-stat updates can add noise to finite differences unless the randomness or model state is controlled.

## Optimizer API

```python
optimizer = CoCD(
    params,
    lr=1e-2
    weight_decay=0.0,
    init_grad=None,
    eps=1e-1,
    compute_budget=1,
    memory_budget=None,
    momentum=1.0,
    central_difference=True,
)
```

| Argument | Meaning |
| --- | --- |
| `params` | Iterable of PyTorch parameters. |
| `lr` | Learning rate used when applying the gradient buffer. |
| `weight_decay` | Decoupled weight decay applied during parameter updates. |
| `init_grad` | Optional initial flattened gradient buffer. Must have `memory_budget` entries. |
| `eps` | Finite-difference perturbation radius. This is also the smoothing radius discussed in the paper. |
| `compute_budget` | Number of scalar coordinates refreshed per optimizer step. The current finite-difference helper evaluates the closure twice per refreshed coordinate, so one step costs `2 * compute_budget` objective evaluations. |
| `memory_budget` | Number of coordinate estimates stored in the FIFO buffer. Defaults to the full parameter count. Smaller values reduce memory use but truncate stale history. |
| `momentum` | Decay applied to the gradient buffer before refreshing coordinates. `1.0` preserves history; `0.0` recovers BCCD without stale-gradient reuse. |
| `central_difference` | Whether to use symmetric central differences. `False` uses a one-sided difference while still recomputing the unperturbed objective for each refreshed coordinate. |

The optimizer also supports `state_dict()`, `load_state_dict()`, and `reset()`.

## Choosing Hyperparameters

CoCD has three main components, and they are not equally important.

First, stale-gradient reuse is the core mechanism. The FIFO buffer provides a dense update signal even when only a small number of coordinates are refreshed at each step. Second, a appropriately larger finite-difference radius `eps` can be essential on rough non-convex landscapes because it implicitly smooths the objective and reduces the effective coordinate-wise Lipschitz constant. Finally, `momentum` is the main safety valve: it discounts the oldest gradient estimates so spatial drift in stale gradients does not dominate the update, leading to better stability.

A practical tuning hierarchy is:

1. **Budgets (`compute_budget` and `memory_budget`)**: maximize `compute_budget` within your acceptable wall-clock cost, and set `memory_budget` as large as your hardware allows. The default `memory_budget=None` stores one estimate per scalar parameter.
2. **Learning rate (`lr`)**: start from the standard first-order SGD learning rate for the architecture.
3. **Smoothing (`eps`)**: start with `eps=1.0`, which was empirically robust in the paper. If the objective scale makes this perturbation too large, reduce it; if tiny finite differences are noisy or unstable, increase it.
4. **Momentum (`momentum`)**: tune this primarily for stability. Start with `momentum=1.0` for full stale-gradient reuse, then gradually decrease it, for example to `0.99`, `0.95`, `0.9`, or lower, until the optimization trajectory stabilizes.


## Checkpointing

Save the model and optimizer state just like standard PyTorch optimizers:

```python
torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    "checkpoint.pt",
)
```

```python
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

## Citation

If this implementation helps your research, please cite:

```bibtex
@inproceedings{liang2026cocd,
  title={Turning Stale Gradients into Stable Gradients: Coherent Coordinate Descent with Implicit Landscape Smoothing for Lightweight Zeroth-Order Optimization},
  author={Liang, Chen and Sun, Xiatao and Wang, Qian and Rakita, Daniel},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning},
  year={2026},
  publisher={PMLR}
}
```

## License

This project is released under the MIT License.
