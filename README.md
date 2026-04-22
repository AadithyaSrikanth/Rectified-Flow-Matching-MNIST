# Minimal MNIST Rectified Flow with Reflow and One-Step Distillation

This repository contains a **minimal unconditional MNIST implementation** of a staged rectified-flow pipeline built around a **Flow Matching baseline**, followed by **reflow** and **one-step distillation**.

The code is centered in `RectFlowMNIST.py` (please refer the google drive link for 'RectFlowMNIST.ipynb') and is designed to be easy to read, modify, and extend. The implementation stays intentionally compact while still covering the main empirical ideas:

- **Stage 1:** Flow Matching baseline with straight-line interpolation
- **Stage 2:** First reflow / rectification pass
- **Optional Stage 3:** Second reflow pass
- **One-step distillation:** Student model trained from a multi-step teacher
- **Low-NFE evaluation:** Qualitative and quantitative comparisons at small numbers of function evaluations

---

## What this implementation includes

### Training pipeline

- A **Flow Matching baseline** trained on MNIST using the linear interpolation
  `z_t = (1 - t) x_0 + t x_1`
  with target velocity
  `v*(z_t, t) = x_1 - x_0`.
- A **reflow stage** trained on teacher-generated endpoint pairs.
- An **optional second reflow stage**.
- **One-step distillation** from the stage-2 teacher and optionally from the stage-3 teacher.

### Evaluation pipeline

- **Euler sampling** for visible low-NFE sample sweeps.
- **Midpoint integration** for teacher endpoint generation and reflow/distillation targets.
- Metric tracking during training and at final evaluation.

### Outputs saved by the implementation

Outputs are written under `mnist_rectified_flow_notebook_runs/`.

Saved artifacts include:
- model checkpoints for stage 1 / stage 2 / optional stage 3
- distillation checkpoints
- per-stage JSON metric summaries
- per-epoch metric histories
- sample grids and comparison figures
- a final bundled summary file

---

## Model architecture

The velocity model is a compact convolutional network with time conditioning:

- input resolution: **28 x 28 x 1**
- hidden width: **128**
- number of layers: **6**
- activation: **SiLU**
- normalization: **GroupNorm** in the residual stack
- trainable parameters: **911,105**

The model is implemented through:
- `ConvBlock`
- `ImageFlow`

Time conditioning is applied through a separate `time_project` branch and injected into the residual convolutional stack.

---

## Dataset and preprocessing

- dataset: **MNIST**
- split used for training: **train split**
- task: **unconditional generation**
- transform: **`ToTensor()` only**
- pixel range after preprocessing: **[0, 1]**
- extra normalization: **none**
- data augmentation: **none**
- source distribution: **standard Gaussian noise**

---

## Default configuration

These are the default settings used in the implementation:

| Setting | Value |
|---|---:|
| Seed | `1234` |
| Batch size | `128` |
| Hidden dim | `128` |
| Layers | `6` |
| Learning rate | `1e-4` |
| Stage-1 epochs | `50` |
| Stage-2 reflow epochs | `20` |
| Distillation epochs | `12` |
| Teacher pair steps | `32` |
| Teacher pair solver | `midpoint` |
| Visible sampling solver | `euler` |
| Sample NFEs | `[1, 2, 4, 8, 16]` |
| Evaluation batch size | `256` |
| Optional stage 3 | `True` |
| Num workers | `0` |

These settings are defined in **Block 1** and can be edited easily.

---

## Solvers and what they are used for

### Euler solver

Euler is used for the **visible low-NFE sampling comparisons**:

`z_{k+1} = z_k + Δt · v_θ(z_k, t_k)`

This is the solver used for the sample sweeps at **1, 2, 4, 8, 16** steps.

### Midpoint solver

Midpoint is used when a **more accurate teacher endpoint** is needed:

- `v_1 = v_θ(z_k, t_k)`
- `z_{k+1/2} = z_k + (1/2) Δt · v_1`
- `v_2 = v_θ(z_{k+1/2}, t_k + (1/2)Δt)`
- `z_{k+1} = z_k + Δt · v_2`

In this repository, midpoint is used for:
- generating **teacher pairs** for reflow
- generating **teacher endpoints** for distillation targets
- computing diagnostics against a higher-quality teacher rollout

---

## Metrics used

The implementation tracks the following metrics.

### 1. Straightness

This measures how close the predicted velocity at an interpolated point is to the straight-line chord between endpoints.

Lower is better.

### 2. Endpoint consistency

This measures whether a local velocity prediction at time `t` extrapolates to the same final endpoint as the numerically integrated trajectory.

Lower is better.

### 3. Quadratic transport cost

This measures the squared displacement magnitude between source noise and generated endpoint.

This is mainly used as a secondary across-stage diagnostic.

### 4. One-step endpoint MSE

For distillation, this measures how well the one-step student matches the endpoint produced by the teacher flow.

Lower is better.

### 5. Naive one-step teacher baseline

The implementation also compares distillation against the naive one-step approximation obtained directly from the teacher at `t = 0`.

---

## Results from the included run

### Stage 1 vs Stage 2

| Model | Straightness ↓ | Endpoint consistency ↓ | Quadratic transport cost |
|---|---:|---:|---:|
| Stage 1 (Flow Matching baseline) | 0.03651 | 0.01779 | 760.65 |
| Stage 2 (first reflow) | 0.00259 | 0.00147 | 752.85 |

**Takeaway:** the first reflow stage substantially improves path straightness and endpoint consistency relative to the Flow Matching baseline, especially for low-NFE sampling.

### Optional Stage 3

| Model | Straightness ↓ | Endpoint consistency ↓ | Quadratic transport cost |
|---|---:|---:|---:|
| Stage 1 | 0.03850 | 0.01759 | 760.51 |
| Stage 2 | 0.00289 | 0.00149 | 752.24 |
| Stage 3 | 0.00178 | 0.00080 | 744.48 |

**Takeaway:** a second reflow stage can further improve straightness and endpoint consistency, though the main qualitative gain is already visible after stage 2.

### Distillation after Stage 2

| Quantity | Value |
|---|---:|
| Stage-2 source-flow straightness | 0.00262 |
| Stage-2 source-flow endpoint consistency | 0.00148 |
| Naive teacher one-step endpoint error | 0.00694 |
| Distilled one-step endpoint MSE | 0.00473 |

**Takeaway:** the one-step distilled student improves over the naive one-step teacher approximation and produces recognizable one-step samples, though multi-step stage-2 sampling remains better visually.

### Optional Distillation after Stage 3

| Quantity | Value |
|---|---:|
| Stage-3 source-flow straightness | 0.00214 |
| Stage-3 source-flow endpoint consistency | 0.00080 |
| Naive teacher one-step endpoint error | 0.00425 |
| Distilled one-step endpoint MSE | 0.00275 |

---

## How to run

### Option 1: Jupyter / Colab

Open the notebook and run cells in order:

```bash
jupyter notebook RectFlowMNIST.ipynb
```

Or upload the notebook to Google Colab.

### Option 2: local environment

Install the main dependencies:

```bash
pip install torch torchvision matplotlib numpy tqdm
```

Then run the implementation end-to-end.

---

## Implementation structure

The code is organized into blocks:

- imports and configuration
- data loading
- model definition
- ODE integration and sampling helpers
- training batch samplers
- metric helpers
- training functions
- one-step distillation helpers
- stage-1 / stage-2 training and main evaluation
- stage-2 one-step distillation
- optional stage 3
- optional stage-3 distillation

---

## Expected outputs

After a full run, you should have:
- trained stage-1 and stage-2 models
- optional stage-3 results
- sample grids across low NFEs
- metric curves over training checkpoints
- one-step distillation comparisons
- JSON summaries and checkpoint bundles for later analysis

---

## Notes

- This is a **minimal educational implementation**, not a heavily optimized large-scale training codebase.
- The code is intentionally compact and implementation-friendly.
- `NUM_WORKERS = 0` is used for stability.
- The most informative visual comparison is typically **Stage 1 vs Stage 2 at low NFE**, not necessarily the highest-NFE sample grid.

---

## Related papers

This implementation is inspired by the following lines of work:
- **Flow Matching for Generative Modeling**
- **Learning to Generate and Transfer Data with Rectified Flow**

Conceptually:
- **Stage 1** is best viewed as a **Flow Matching baseline with straight-line interpolation**.
- **Stage 2 / Stage 3** correspond to **reflow / rectification**.
- **One-step distillation** compresses a multi-step source flow into a single update.

---
