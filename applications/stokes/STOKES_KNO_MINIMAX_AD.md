# Stokes KNO Minimax Notes

Current active script: `stokes_kno_minimax_ad.py`

## Goal

Learn an operator from Fourier inflow coefficients to the full Stokes velocity field, while using the original solver from `fourier_inflow_ad.py` as the ground-truth PDE model.

The solver is not `jit`-compatible, so the code splits the computation into:

- non-jitted PDE forward solves and adjoint VJPs
- jitted KNO/operator losses and optimizer updates where possible

## Final operator choice

We are not using the earlier GP-neurop or FNO paths in the active setup.

The active operator is a `kno_eqx`-style boundary-to-field KNO:

- input: boundary inflow values sampled on the actual inflow boundary nodes
- output: velocity on the full velocity node grid

Mathematically:

1. Fourier coefficients `c in R^m` define the inflow trace

   `g_c(y) = sum_k c_k sin(pi k y)`

2. This trace is sampled on the inflow boundary nodes.

3. The KNO learns a map

   `g_c|_{boundary} -> u(x)`

   where `u(x)` is the full 2D velocity field on the FEM velocity grid.

Programmatically:

- the boundary nodes come from `solver.problem.fe_u.node_inds_list[0]`
- the full output grid comes from `solver.problem.fe_u.points`
- the two velocity components are predicted by two scalar KNO heads and concatenated

## Boundary quadrature

We needed a quadrature rule on the boundary nodes because the KNO integration layers integrate over the boundary trace.

Current rule:

- sort the inflow boundary points by their boundary coordinate
- use a trapezoid-style rule based on neighboring edge lengths

Mathematically this approximates

`int_{boundary} k(s, t) f(t) dt`

with a weighted sum over boundary nodes.

This is implemented in `boundary_trapezoid_rule(...)`.

## Kernel choice

For now the cheap integration/interpolation kernel is Mat\'ern-C2.

Important implementation detail:

- do not build the Gramian with a double `vmap`
- use the dense pairwise-distance formula

Current form:

`r_ij^2 = ||x_i||^2 + ||y_j||^2 - 2 x_i^T y_j`

then

`k(x_i, y_j) = (1 + sqrt(5) r + (5/3) r^2) exp(-sqrt(5) r)`

This is cheaper and cleaner for the boundary and output interpolation solves.

## KNO head used in the script

The script uses a local stabilized adaptation of the `kno_eqx` KNO triangle logic.

Reasons:

- direct reuse was numerically brittle here
- we needed explicit control of interpolation jitter
- we needed to freeze geometry arrays cleanly

The head does:

1. lift boundary samples together with boundary coordinates
2. interpolate from boundary nodes to boundary quadrature nodes with an input kernel solve
3. apply kernel integral updates plus pointwise `1x1` conv skips in latent space
4. project latent values back to scalars on the quadrature nodes
5. interpolate from quadrature nodes to the full velocity grid with an output kernel solve

## Numerical stabilization

The script now explicitly stabilizes kernel solves:

- symmetrize each Gramian as `0.5 * (K + K.T)`
- add `kernel_jitter * I`

This matters mathematically because the interpolation/integration steps assume positive-definite Gram matrices. Without stabilization, the operator update could produce non-finite values.

## Important bug that was fixed

At one point the operator optimizer was updating fixed geometry arrays:

- boundary locations
- boundary quadrature weights
- boundary basis matrix
- output grid locations

That is wrong mathematically because those arrays define the discretization, not trainable parameters.

Programmatically this caused `nan`s after operator steps.

Fix:

- build an explicit operator filter
- train only actual KNO parameters
- freeze all geometry/discretization arrays

## Training objective

The model is a minimax game:

`min_theta max_phi E_eps[ relL2( u(s_phi(eps)), G_theta(s_phi(eps)) ) ]`

where:

- `s_phi` is the Gaussian sampler over Fourier coefficients
- `u(c)` is the original Stokes solver output
- `G_theta(c)` is the KNO prediction

The relative error is

`||u_true - u_pred||_2 / (||u_true||_2 + 1e-12)`

## Sampler gradient

The sampler update keeps the exact chain rule split:

`dL/dc = (dL/dc)|operator + J_solver(c)^T (dL/du_true)`

Meaning:

- one part comes from how changing coefficients changes the KNO input directly
- the other part comes from how changing coefficients changes the true PDE solution

Because the original solver is not jittable, this PDE sensitivity is obtained outside `jit` via solver VJPs / adjoints.

## Current epoch schedule

We changed the inner-loop semantics.

Now each epoch:

1. samples one latent batch `eps_batch`
2. reuses that same batch for all `gen_steps`
3. after generator updates, forms one operator batch from the current sampler
4. solves the true PDE once for that operator batch
5. reuses that fixed operator batch for all `op_steps`

So `gen-steps` and `op-steps` are numbers of optimizer updates on a fixed sampled batch, not numbers of resampled batches.

## Batch arguments

There is now a single

- `--batch`

instead of separate `--gen-batch` and `--op-batch`, because both phases now use the same epoch-level sampled batch.

## Quiet solver flag

Added:

- `--quiet-solver`

This suppresses:

- `jax_fem` logger output
- solver forward-solve chatter
- adjoint-solve chatter
- import-time banner noise from the solver path

So you can still see the high-level epoch metrics without the FEM spam.

## Evaluation criterion

The main evaluation idea we settled on is:

- fix a held-out set of coefficient samples once
- compute their true Stokes solutions once
- compare models on the exact same held-out set

For held-out coefficients `{c_i}` with truths `{u_i}`, define

`e_i = relL2(u_i, G(c_i))`

Then track:

- held-out mean error: `mean_i e_i`
- held-out worst-case error: `max_i e_i`

This is the key minimax metric: the adversarially trained model should ideally have lower held-out worst-case error than a model trained with the generator frozen.

Current script flags:

- `--heldout-batch`
- `--heldout-seed`
- `--freeze-generator`

`--freeze-generator` gives the baseline where the operator is trained while the sampler stays fixed.

## Comparison to run

Recommended fair comparison:

1. adversarial/minimax:

   `python stokes_kno_minimax_ad.py --quiet-solver ...`

2. frozen-generator baseline:

   `python stokes_kno_minimax_ad.py --quiet-solver --freeze-generator ...`

Use the same:

- `--seed`
- `--heldout-seed`
- `--batch`
- `--epochs`
- `--op-steps`
- model hyperparameters

Then compare:

- `heldout_mean_rel_l2`
- `heldout_worst_rel_l2`
- especially `heldout_worst_rel_l2`

## Current printed metrics

Each epoch currently prints:

- `gen_rel_l2`
- `op_rel_l2`
- `heldout_mean_rel_l2`
- `heldout_worst_rel_l2`
- `heldout_worst_best`
- `sig_mean`

Interpretation:

- `gen_rel_l2`: adversarial objective on the generator batch
- `op_rel_l2`: operator fit on the current operator batch
- `heldout_mean_rel_l2`: average out-of-sample error on the fixed held-out set
- `heldout_worst_rel_l2`: worst out-of-sample error on the fixed held-out set
- `heldout_worst_best`: best worst-case value seen so far
- `sig_mean`: average sampler standard deviation

## Useful example command

```bash
python stokes_kno_minimax_ad.py \
  --quiet-solver \
  --epochs 10 \
  --gen-steps 1 \
  --op-steps 100 \
  --batch 1 \
  --heldout-batch 16 \
  --num-modes 2
```

## Files of interest

- `stokes_kno_minimax_ad.py`: current active implementation
- `fourier_inflow_ad.py`: original Stokes solver used as teacher
- `stokes_gp_neurop_minimax_ad.py`: earlier GP-neurop-based exploratory path, not the active one

