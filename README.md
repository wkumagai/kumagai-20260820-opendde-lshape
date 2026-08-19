# OpenDDE fine-tuning — restoring L_shape

Forked from `auto-res2/matsuzawa-20260815-opendde-multinode-longrun-3` (Matsuzawa).
His repository is his experimental record and is not modified here.

It adds the shape-complementarity term to the fine-tuning objective, on top
of the W&B logging and intermediate checkpoints this line of work already
carried:

- **W&B logging.** `config.yaml` carried a `wandb:` block that no code read.
  Rank 0 alone opens the run — sixteen ranks calling `wandb.init()` would file
  sixteen runs for one job — and per-epoch loss and gradient norm are
  rank-averaged before being logged. A logging failure warns and continues; it
  never takes a 16-GPU job down.
- **Intermediate checkpoints.** `--checkpoint-at "1,2,4,8,16,32"` writes weights
  at those epochs. Previously the only checkpoint was the final one, so a long
  run said nothing about performance until it ended. Each file is a complete
  state_dict, so scoring an epoch sweep is several ordinary evaluations rather
  than a new code path.

`epochs` in `config.yaml` raises the epoch count without redefining what
`mode: full` means for runs that already claimed it.

## Environment

`wandb` is not installed in the shared OpenDDE venv, and that venv belongs to
Muto — it is read, never written. Instead `wandb` lives in
`/data1/rkp00041/rku00121/pylibs` and reaches the interpreter through
`PYTHONPATH`, which `src/main.py` already forwards.

## The restored term

The harness optimised one loss, `diffusion_loss` — AF3's weighted diffusion MSE,
`L_diff`. OpenDDE's technical report (arXiv:2607.03787, Section A.2.1) gives its
geometric objective as

```
L_geom = L_diff + λ_shape·L_shape + λ_local·L_local + λ_sc/base·L_sc/base + λ_χ·L_χ
```

so we were fine-tuning released weights under a strictly weaker objective than
the one they were pre-trained with. `L_shape` is restored here; the other three
are not, deliberately, because adding them together would leave no way to tell
which one moved a number.

```
L_shape = λ_p·Huber(q_uv, q*_uv) + λ_t·Huber(q_u, q*_u) + λ_g·Huber(q_global, q*_global)
```

`q_uv` is a cross-chain token-pair score — the product of a face-to-face factor,
an opposed-normal factor, a gap factor and an anti-clash factor, evaluated only
on pairs from different chains that are inside the interface cutoff. `q_u` is its
per-token pool, `q_global` the structure-level pool. Predicted and ground-truth
coordinates go through the identical field and are compared with a Huber, so
the term measures reproduction of the native interface rather than contact for
its own sake.

Nothing here is reimplemented. The field is
`opendde/model/shape_complementarity.py` from the released build, which already
runs at inference time via `add_shape_complementarity_predictions`; the three
sub-weights (0.4 / 1.0 / 0.4) and every geometry constant come out of
`confidence.shape_comp` in `opendde/config/model_base.py` and are read from the
instantiated config at run time. `λ_shape` is the one number the report does not
publish, so it is `shape_loss.lambda_shape` in `config.yaml`.

### What we checked before spending GPU time

The field detaches its own summary statistics and runs parts of itself under
`torch.no_grad`, so whether a gradient survives to the coordinates is a question
and not an assumption. Running the released field verbatim on a toy two-chain
system:

- `q_uv`, `q_u` and `q_global` all carry gradient to the input coordinates.
  Only `shape_comp_pair_mean`, `shape_comp_pair_topk_mean` and
  `shape_comp_valid_pair_frac` are detached, and `L_shape` uses none of them.
- The term is rigid-invariant to 1e-8, so ground truth needs no Kabsch
  alignment and goes in as the augmented coordinates already are.
- It is exactly zero when the prediction equals the truth, and exactly zero for
  a single-chain system.
- **It is a near-field loss.** Its gradient is largest at small structural
  error and collapses as error grows — on the toy system, `|∂L/∂x|` fell from
  1e-2 at 0.5 Å to 4e-6 at 1.5 Å and to exactly zero at 5 Å, while the loss
  value itself plateaued. The `relu` in the face-to-face factor is what does
  this: two surfaces that no longer face each other contribute nothing. It
  cannot pull distant chains together, and the anti-clash factor suppresses
  pairs closer than `clash_distance`.

### Noise levels

The training schedule's median σ is 4.8 Å (mean displacement 7.7 Å) and its top
16% is above 21.6 Å. At those levels no interface exists to be complementary,
and the finding above says the term contributes a constant with no gradient
there. Each sample is therefore weighted by EDM's own skip coefficient,
`c_skip = σ_data²/(σ² + σ_data²)` — 0.92 at the median, 0.35 at the top-16%
level, falling as 1/σ² above that. It is continuous and it is already the
quantity the diffusion module uses to say how much of its input survived, which
a hand-picked cutoff is not. `shape_loss.sigma_max` adds a hard cutoff on top
for an explicit ablation and is off by default.

### Masks

Pair and token masks are the **union** of the predicted and ground-truth ones.
Masking on the prediction alone would let the model escape the term by pulling
the chains apart until no cross-chain pair is inside the cutoff; masking on the
truth alone cannot see a contact the model invented. The union does make the set
of measured residuals depend on the ground truth, which is stated here rather
than hidden.

### Control

`shape_loss.lambda_shape: 0.0` does not merely multiply the term by zero — it
stops it being evaluated at all, so the run draws the same random numbers and
builds the same graph as the diffusion-only harness. That setting is the
control arm.

### Logging

`train/loss_total`, `train/loss_diff` and `train/loss_shape` are logged
separately, along with `train/shape_pair`, `train/shape_token`,
`train/shape_global`, `train/shape_sigma_weight` and `train/shape_samples`. A
total alone cannot say which term moved. The first step of each process also
prints `SHAPE_GRAD_CHECK` with the measured gradient norm of the shape term with
respect to the denoised coordinates, so a silently-constant term is visible in
the log rather than inferred.
