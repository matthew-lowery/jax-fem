from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import os
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as onp
import optax

from div_stuff.rbf_fd_divergence import build_rbffd_divergence
from local_kno_utils import create_lifted_module as clm
from training_metric_plots import save_training_metrics_figure
from training_wandb import finish_wandb_run, init_wandb_run, log_wandb_metrics


@contextlib.contextmanager
def maybe_quiet_solver(quiet):
    if not quiet:
        yield
        return

    logger = logging.getLogger("jax_fem")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
    finally:
        logger.setLevel(old_level)


def load_build_solver(quiet):
    with maybe_quiet_solver(quiet):
        module = importlib.import_module("fourier_inflow_ad")
    return module.build_solver


def rel_l2(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return jnp.linalg.norm(y_true - y_pred) / (jnp.linalg.norm(y_true) + 1e-12)


class GaussianSampler(eqx.Module):
    mu: jax.Array
    raw_sig: jax.Array

    def __init__(self, *, dim, key):
        key_mu, key_sig = jr.split(key)
        self.mu = 0.01 * jr.normal(key_mu, (dim,))
        self.raw_sig = -2.0 + 0.1 * jr.normal(key_sig, (dim,))

    @property
    def sig(self):
        return jax.nn.softplus(self.raw_sig)

    def sample_eps(self, eps):
        return self.mu + eps * self.sig

    def rvs(self, key, shape):
        eps = jr.normal(key, shape + self.mu.shape)
        return self.sample_eps(eps)


class MaternC2Kernel(eqx.Module):
    scale: jax.Array

    def __init__(self, *, key):
        key, _ = jr.split(key)
        self.scale = jr.uniform(key, minval=-4.5, maxval=-3.5)

    def __call__(self, x, y):
        scale = jax.nn.softplus(self.scale)
        r = jnp.sum(x**2, axis=1)[:, None] + jnp.sum(y**2, axis=1)[None, :] - 2 * x @ y.T
        r = jnp.sqrt(jnp.maximum(r, 0.0))
        r_scaled = r * scale
        sq5 = jnp.sqrt(5.0)
        return (1.0 + sq5 * r_scaled + (5.0 / 3.0) * r_scaled**2) * jnp.exp(-sq5 * r_scaled)


class BoundaryKNOHead(eqx.Module):
    input_kernel: eqx.Module
    output_kernel: eqx.Module
    integration_kernels: list
    proj_kernels: list
    pointwise_layers: list
    lift_kernel: eqx.Module
    lift_dim: int
    depth: int
    kernel_jitter: float

    def __init__(
        self,
        *,
        input_kernel,
        output_kernel,
        integration_kernel,
        lift_dim,
        depth,
        in_feats,
        kernel_jitter,
        key,
    ):
        keys = jr.split(key, 2)
        self.integration_kernels = [clm(integration_kernel, lift_dim=lift_dim, key=k) for k in jr.split(keys[0], depth)]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=k) for k in jr.split(keys[1], depth)]

        keys = jr.split(keys[0], 4)
        self.proj_kernels = [
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]),
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]),
            eqx.nn.Linear(lift_dim, 1, key=keys[2]),
        ]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[3])

        keys = jr.split(keys[0], 2)
        self.input_kernel = input_kernel(key=keys[0])
        self.output_kernel = output_kernel(key=keys[1])

        self.lift_dim = lift_dim
        self.depth = depth
        self.kernel_jitter = kernel_jitter

    def stabilized_kernel(self, kernel, locs):
        eye = jnp.eye(len(locs), dtype=locs.dtype)
        gram = kernel(locs, locs)
        gram = 0.5 * (gram + gram.T)
        return gram + self.kernel_jitter * eye

    def __call__(self, f_x, x_grid, y_grid, q, w):
        def integration_transform(int_kernel, f_q):
            gram = int_kernel(q, q)
            gram = 0.5 * (gram + gram.T)
            return gram @ (f_q * w[:, 0])

        f_x = jnp.concatenate((f_x, x_grid), axis=-1)
        f_x = eqx.filter_vmap(self.lift_kernel)(f_x).reshape(len(x_grid), self.lift_dim)

        k_xx = self.stabilized_kernel(self.input_kernel, x_grid)
        k_xq = self.input_kernel(x_grid, q)
        f_q = jnp.einsum("mc,qm->qc", f_x, jnp.linalg.solve(k_xx, k_xq).T)
        f_q = jax.nn.gelu(f_q)

        for i in range(self.depth - 1):
            f_q_skip = self.pointwise_layers[i](f_q.T).T
            f_q = eqx.filter_vmap(
                lambda int_kernel, current_f: integration_transform(int_kernel, current_f),
                in_axes=(eqx.if_array(0), 1),
                out_axes=1,
            )(self.integration_kernels[i], f_q)
            f_q = jax.nn.gelu(f_q_skip + f_q)

        f_q_skip = self.pointwise_layers[-1](f_q.T).T
        f_q = eqx.filter_vmap(
            lambda int_kernel, current_f: integration_transform(int_kernel, current_f),
            in_axes=(eqx.if_array(0), 1),
            out_axes=1,
        )(self.integration_kernels[-1], f_q)
        f_q = f_q_skip + f_q

        f_q = f_q.reshape(-1, self.lift_dim)
        f_q = jax.nn.gelu(eqx.filter_vmap(self.proj_kernels[0])(f_q))
        f_q = jax.nn.gelu(eqx.filter_vmap(self.proj_kernels[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_kernels[2])(f_q).reshape(len(q), 1)

        k_qq = self.stabilized_kernel(self.output_kernel, q)
        k_qy = self.output_kernel(q, y_grid)
        return jnp.einsum("mc,qm->qc", f_q, jnp.linalg.solve(k_qq, k_qy).T)


def boundary_trapezoid_rule(boundary_locs):
    if len(boundary_locs) == 1:
        return jnp.ones((1, 1))
    edge_lengths = jnp.linalg.norm(boundary_locs[1:] - boundary_locs[:-1], axis=1)
    weights = jnp.zeros((len(boundary_locs),))
    weights = weights.at[0].set(edge_lengths[0] / 2.0)
    weights = weights.at[-1].set(edge_lengths[-1] / 2.0)
    weights = weights.at[1:-1].set((edge_lengths[:-1] + edge_lengths[1:]) / 2.0)
    return weights[:, None]


def boundary_face_quadrature(problem, inflow_location_fn):
    boundary_inds = problem.fe_u.get_boundary_conditions_inds([inflow_location_fn])[0]
    quad_locs = onp.asarray(problem.fe_u.get_physical_surface_quad_points(boundary_inds)).reshape(-1, problem.dim)
    _, quad_weights = problem.fe_u.get_face_shape_grads(boundary_inds)
    quad_weights = onp.asarray(quad_weights).reshape(-1)
    order = onp.lexsort((quad_locs[:, 0], quad_locs[:, 1]))
    quad_locs = jnp.asarray(quad_locs[order])
    quad_weights = jnp.asarray(quad_weights[order])[:, None]
    return quad_locs, quad_weights


def build_boundary_data(problem, num_modes, quadrature_rule):
    boundary_node_inds = onp.asarray(problem.fe_u.node_inds_list[0])
    boundary_node_inds = onp.unique(boundary_node_inds)
    boundary_locs = onp.asarray(problem.fe_u.points[boundary_node_inds])
    order = onp.lexsort((boundary_locs[:, 0], boundary_locs[:, 1]))
    boundary_locs = jnp.asarray(boundary_locs[order])
    modes = jnp.arange(1, num_modes + 1, dtype=boundary_locs.dtype)
    boundary_basis = jnp.sin(jnp.pi * boundary_locs[:, 1:2] * modes[None, :])
    inflow_location_fn = problem.fe_u.dirichlet_bc_info[0][0]

    if quadrature_rule == "trapezoid":
        quadrature_locs = boundary_locs
        quadrature_w = boundary_trapezoid_rule(boundary_locs)
    elif quadrature_rule == "fem-face":
        quadrature_locs, quadrature_w = boundary_face_quadrature(problem, inflow_location_fn)
    else:
        raise ValueError(f"Unknown boundary quadrature rule: {quadrature_rule}")

    return boundary_locs, quadrature_locs, quadrature_w, boundary_basis


class VectorBoundaryKNO(eqx.Module):
    heads: tuple[eqx.Module, eqx.Module]
    boundary_locs: jax.Array
    quadrature_locs: jax.Array
    quadrature_w: jax.Array
    boundary_basis: jax.Array
    output_locs: jax.Array

    def boundary_values(self, coefs):
        return self.boundary_basis @ coefs

    def __call__(self, coefs):
        boundary_vals = self.boundary_values(coefs)[:, None]
        outputs = [
            head(
                f_x=boundary_vals,
                x_grid=self.boundary_locs,
                y_grid=self.output_locs,
                q=self.quadrature_locs,
                w=self.quadrature_w,
            )
            for head in self.heads
        ]
        return jnp.concatenate(outputs, axis=-1)


class StokesKNOMinimaxModel(eqx.Module):
    sampler: eqx.Module
    operator: eqx.Module

    def predict(self, coefs):
        return self.operator(coefs)


def build_operator_filter(operator):
    operator_filter = jax.tree.map(eqx.is_array, operator)
    operator_filter = eqx.tree_at(lambda current: current.boundary_locs, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.quadrature_locs, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.quadrature_w, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.boundary_basis, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.output_locs, operator_filter, replace=False)
    return operator_filter


def build_model(
    *,
    build_solver_fn,
    num_modes,
    lift_dim,
    depth,
    kernel_jitter,
    boundary_quadrature_rule,
    quiet_solver,
    key,
):
    key_sampler, key_operator = jr.split(key)
    with maybe_quiet_solver(quiet_solver):
        solver = build_solver_fn(num_modes=num_modes)
    boundary_locs, quadrature_locs, quadrature_w, boundary_basis = build_boundary_data(
        solver.problem,
        num_modes,
        boundary_quadrature_rule,
    )
    output_locs = jnp.asarray(solver.problem.fe_u.points)

    kernel_factory = MaternC2Kernel
    operator_keys = jr.split(key_operator, 2)
    heads = tuple(
        BoundaryKNOHead(
            input_kernel=kernel_factory,
            output_kernel=kernel_factory,
            integration_kernel=kernel_factory,
            lift_dim=lift_dim,
            depth=depth,
            in_feats=boundary_locs.shape[1] + 1,
            kernel_jitter=kernel_jitter,
            key=current_key,
        )
        for current_key in operator_keys
    )

    model = StokesKNOMinimaxModel(
        sampler=GaussianSampler(dim=num_modes, key=key_sampler),
        operator=VectorBoundaryKNO(
            heads=heads,
            boundary_locs=boundary_locs,
            quadrature_locs=quadrature_locs,
            quadrature_w=quadrature_w,
            boundary_basis=boundary_basis,
            output_locs=output_locs,
        ),
    )
    return model, solver


@eqx.filter_jit
def sample_coefs_batch(sampler, eps_batch):
    return jax.vmap(sampler.sample_eps)(eps_batch)


@eqx.filter_jit
def mean_batch_loss(model, coefs_batch, u_true_batch):
    losses = jax.vmap(lambda coefs, u_true: rel_l2(u_true, model.predict(coefs)))(
        coefs_batch,
        u_true_batch,
    )
    return losses.mean()


@eqx.filter_jit
def loss_input_partials(model, coefs_batch, u_true_batch):
    def one_sample(coefs, u_true):
        def loss_fn(current_coefs, current_u_true):
            return rel_l2(current_u_true, model.predict(current_coefs))

        loss, (grad_coefs, grad_u_true) = jax.value_and_grad(loss_fn, argnums=(0, 1))(coefs, u_true)
        return loss, grad_coefs, grad_u_true

    losses, grad_coefs, grad_u_true = jax.vmap(one_sample)(coefs_batch, u_true_batch)
    return losses.mean(), grad_coefs, grad_u_true


@eqx.filter_jit
def eval_metrics(model, coefs_batch, u_true_batch):
    rel_errors = jax.vmap(lambda coefs, u_true: rel_l2(u_true, model.predict(coefs)))(
        coefs_batch,
        u_true_batch,
    )
    return rel_errors.mean(), rel_errors.max()


@eqx.filter_jit
def pred_div_metrics(model, coefs_batch, divergence_operator, divergence_indices):
    u_pred_batch = jax.vmap(model.predict)(coefs_batch)
    return pred_div_metrics_from_u_pred(u_pred_batch, divergence_operator, divergence_indices)


@eqx.filter_jit
def pred_div_metrics_from_u_pred(u_pred_batch, divergence_operator, divergence_indices):
    div_batch = jnp.take(divergence_operator(u_pred_batch), divergence_indices, axis=1)
    mean_abs_div = jnp.mean(jnp.abs(div_batch), axis=1)
    return mean_abs_div.mean(), mean_abs_div.max()


@eqx.filter_jit
def operator_loss_terms(model, coefs_batch, u_true_batch, divergence_operator, divergence_indices):
    u_pred_batch = jax.vmap(model.predict)(coefs_batch)
    rel_errors = jax.vmap(rel_l2)(u_true_batch, u_pred_batch)
    rel_loss = rel_errors.mean()
    div_loss = jnp.array(0.0, dtype=u_pred_batch.dtype)
    if divergence_operator is not None:
        div_loss, _ = pred_div_metrics_from_u_pred(u_pred_batch, divergence_operator, divergence_indices)
    return rel_loss, div_loss


def solve_batch(solver, coefs_batch, quiet_solver=False):
    with maybe_quiet_solver(quiet_solver):
        return jnp.stack([solver(coefs) for coefs in coefs_batch])


def solve_batch_with_vjps(solver, coefs_batch, quiet_solver=False):
    with maybe_quiet_solver(quiet_solver):
        u_true_batch = []
        solver_vjps = []
        for coefs in coefs_batch:
            u_true, solver_vjp = jax.vjp(solver, coefs)
            u_true_batch.append(u_true)
            solver_vjps.append(solver_vjp)
    return jnp.stack(u_true_batch), solver_vjps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--lift-dim", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--kernel-jitter", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--op-steps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heldout-batch", type=int, default=16)
    parser.add_argument("--heldout-seed", type=int, default=123)
    parser.add_argument("--lr-gen", type=float, default=1e-3)
    parser.add_argument("--lr-op", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-solver", action="store_true")
    parser.add_argument("--freeze-generator", action="store_true")
    parser.add_argument("--generator-objective", choices=("rel-l2", "pred-div"), default="rel-l2")
    parser.add_argument("--op-div-weight", type=float, default=0.0)
    parser.add_argument("--pred-div-xi", type=int, default=3)
    parser.add_argument("--pred-div-chunk-size", type=int, default=512)
    parser.add_argument("--pred-div-include-boundary", action="store_true")
    parser.add_argument("--boundary-quadrature-rule", choices=("trapezoid", "fem-face"), default="trapezoid")
    parser.add_argument("--metrics-plot-path", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="stokes_kno_minimax")
    parser.add_argument("--wandb-name", type=str, default="")
    args = parser.parse_args()

    key = jr.PRNGKey(args.seed)
    build_solver_fn = load_build_solver(args.quiet_solver)
    model, solver = build_model(
        build_solver_fn=build_solver_fn,
        num_modes=args.num_modes,
        lift_dim=args.lift_dim,
        depth=args.depth,
        kernel_jitter=args.kernel_jitter,
        boundary_quadrature_rule=args.boundary_quadrature_rule,
        quiet_solver=args.quiet_solver,
        key=key,
    )

    optimizer_sampler = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_gen))
    optimizer_operator = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_op))
    opt_state_sampler = optimizer_sampler.init(eqx.filter(model.sampler, eqx.is_array))
    operator_filter = build_operator_filter(model.operator)
    opt_state_operator = optimizer_operator.init(eqx.filter(model.operator, operator_filter))
    divergence_operator = None
    divergence_indices = None
    uses_pred_div = args.generator_objective == "pred-div" or args.op_div_weight > 0.0
    if uses_pred_div:
        divergence_operator = build_rbffd_divergence(
            model.operator.output_locs,
            args.pred_div_xi,
            neighbor_chunk_size=args.pred_div_chunk_size,
        )
        divergence_indices = (
            jnp.arange(model.operator.output_locs.shape[0], dtype=jnp.int32)
            if args.pred_div_include_boundary
            else jnp.asarray(onp.flatnonzero(~onp.asarray(divergence_operator.boundary_mask)), dtype=jnp.int32)
        )
    heldout_eps = jr.normal(jr.PRNGKey(args.heldout_seed), (args.heldout_batch, args.num_modes))
    heldout_coefs = sample_coefs_batch(model.sampler, heldout_eps)
    heldout_u_true = solve_batch(solver, heldout_coefs, quiet_solver=args.quiet_solver)

    @eqx.filter_jit
    def train_step_operator(model, opt_state, coefs_batch, u_true_batch):
        operator_params, operator_static = eqx.partition(model.operator, operator_filter)

        def loss_fn(current_params):
            current_operator = eqx.combine(current_params, operator_static)
            current_model = eqx.tree_at(lambda current_model: current_model.operator, model, current_operator)
            rel_loss, div_loss = operator_loss_terms(
                current_model,
                coefs_batch,
                u_true_batch,
                divergence_operator,
                divergence_indices,
            )
            total_loss = rel_loss + args.op_div_weight * div_loss
            return total_loss, (rel_loss, div_loss)

        (loss, (rel_loss, div_loss)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(operator_params)
        updates, opt_state = optimizer_operator.update(grads, opt_state, operator_params)
        operator_params = eqx.apply_updates(operator_params, updates)
        operator = eqx.combine(operator_params, operator_static)
        model = eqx.tree_at(lambda current_model: current_model.operator, model, operator)
        return model, opt_state, loss, rel_loss, div_loss

    def train_step_sampler(model, opt_state, eps_batch, u_true_batch, solver_vjps):
        sampler_params, sampler_static = eqx.partition(model.sampler, eqx.is_array)

        def sample_from_params(current_params):
            current_sampler = eqx.combine(current_params, sampler_static)
            return jax.vmap(current_sampler.sample_eps)(eps_batch)

        coefs_batch, sampler_vjp = jax.vjp(sample_from_params, sampler_params)
        loss, grad_coefs_op, grad_u_true = loss_input_partials(model, coefs_batch, u_true_batch)

        with maybe_quiet_solver(args.quiet_solver):
            grad_coefs_solver = jnp.stack(
                [solver_vjp(current_grad_u_true)[0] for solver_vjp, current_grad_u_true in zip(solver_vjps, grad_u_true)]
            )
        cotangent = -(grad_coefs_op + grad_coefs_solver) / coefs_batch.shape[0]
        sampler_grads = sampler_vjp(cotangent)[0]

        updates, opt_state = optimizer_sampler.update(sampler_grads, opt_state, sampler_params)
        sampler_params = eqx.apply_updates(sampler_params, updates)
        sampler = eqx.combine(sampler_params, sampler_static)
        model = eqx.tree_at(lambda current_model: current_model.sampler, model, sampler)
        return model, opt_state, loss

    @eqx.filter_jit
    def train_step_sampler_pred_div(model, opt_state, eps_batch):
        sampler_params, sampler_static = eqx.partition(model.sampler, eqx.is_array)

        def mean_mean_abs_div_fn(current_params):
            current_sampler = eqx.combine(current_params, sampler_static)
            current_model = eqx.tree_at(lambda current_model: current_model.sampler, model, current_sampler)
            coefs_batch = jax.vmap(current_sampler.sample_eps)(eps_batch)
            mean_abs_div, _ = pred_div_metrics(current_model, coefs_batch, divergence_operator, divergence_indices)
            return mean_abs_div

        # Optax applies gradient descent, so negate the generator objective to perform ascent.
        ascent_surrogate, grads = eqx.filter_value_and_grad(
            lambda current_params: -mean_mean_abs_div_fn(current_params)
        )(sampler_params)
        mean_abs_div = -ascent_surrogate
        updates, opt_state = optimizer_sampler.update(grads, opt_state, sampler_params)
        sampler_params = eqx.apply_updates(sampler_params, updates)
        sampler = eqx.combine(sampler_params, sampler_static)
        model = eqx.tree_at(lambda current_model: current_model.sampler, model, sampler)
        return model, opt_state, mean_abs_div

    heldout_worst_best = jnp.inf
    heldout_div_worst_best = jnp.inf
    gen_metric_label = "gen_rel_l2" if args.generator_objective == "rel-l2" else "gen_pred_div_meanabs"
    op_metric_label = "op_rel_l2" if args.op_div_weight == 0.0 else "op_total_loss"
    history = {
        "epoch": [],
        "gen_loss": [],
        "op_loss": [],
        "op_rel_loss": [],
        "op_div_loss": [],
        "heldout_mean_rel_l2": [],
        "heldout_worst_rel_l2": [],
        "heldout_worst_rel_l2_best": [],
        "heldout_mean_pred_div_meanabs": [],
        "heldout_worst_pred_div_meanabs": [],
        "heldout_worst_pred_div_meanabs_best": [],
        "sig_mean": [],
    }

    startup_metrics = {
        "boundary_nodes": int(model.operator.boundary_locs.shape[0]),
        "boundary_quad_points": int(model.operator.quadrature_locs.shape[0]),
        "velocity_nodes": int(model.operator.output_locs.shape[0]),
        "sampler_dim": int(args.num_modes),
        "generator_objective": args.generator_objective,
        "op_div_weight": float(args.op_div_weight),
        "boundary_quadrature_rule": args.boundary_quadrature_rule,
    }
    print(
        f"boundary_nodes={startup_metrics['boundary_nodes']}, "
        f"boundary_quad_points={startup_metrics['boundary_quad_points']}, "
        f"velocity_nodes={startup_metrics['velocity_nodes']}, "
        f"sampler_dim={startup_metrics['sampler_dim']}, "
        f"generator_objective={startup_metrics['generator_objective']}, "
        f"op_div_weight={startup_metrics['op_div_weight']}, "
        f"boundary_quadrature_rule={startup_metrics['boundary_quadrature_rule']}"
    )
    if divergence_operator is not None:
        div_region = "all" if args.pred_div_include_boundary else "interior"
        startup_metrics.update(
            {
                "pred_div_nodes": int(divergence_operator.x.shape[0]),
                "pred_div_stencil": int(divergence_operator.stencil_size),
                "pred_div_region": div_region,
            }
        )
        print(
            f"pred_div_nodes={divergence_operator.x.shape[0]}, "
            f"pred_div_stencil={divergence_operator.stencil_size}, "
            f"pred_div_region={div_region}"
        )
    wandb_static_metrics = {
        key: value for key, value in startup_metrics.items() if isinstance(value, (int, float, bool))
    }
    wandb_run = (
        init_wandb_run(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            static_metrics=wandb_static_metrics,
        )
        if args.wandb
        else None
    )
    for epoch in range(args.epochs):
        gen_loss = jnp.nan
        op_loss = jnp.nan
        op_rel_loss = jnp.nan
        op_div_loss = jnp.nan
        key, subkey = jr.split(key)
        eps_batch = jr.normal(subkey, (args.batch, args.num_modes))
        coefs_batch = sample_coefs_batch(model.sampler, eps_batch)
        if args.freeze_generator or args.generator_objective == "pred-div":
            u_true_batch = solve_batch(solver, coefs_batch, quiet_solver=args.quiet_solver)
            solver_vjps = None
        else:
            u_true_batch, solver_vjps = solve_batch_with_vjps(solver, coefs_batch, quiet_solver=args.quiet_solver)
        for _ in range(args.op_steps):
            model, opt_state_operator, op_loss, op_rel_loss, op_div_loss = train_step_operator(
                model,
                opt_state_operator,
                coefs_batch,
                u_true_batch,
            )

        if args.freeze_generator:
            if args.generator_objective == "pred-div":
                gen_loss, _ = pred_div_metrics(model, coefs_batch, divergence_operator, divergence_indices)
            else:
                gen_loss = mean_batch_loss(model, coefs_batch, u_true_batch)
        else:
            if args.generator_objective == "pred-div":
                model, opt_state_sampler, gen_loss = train_step_sampler_pred_div(model, opt_state_sampler, eps_batch)
            else:
                model, opt_state_sampler, gen_loss = train_step_sampler(
                    model,
                    opt_state_sampler,
                    eps_batch,
                    u_true_batch,
                    solver_vjps,
                )

        heldout_mean, heldout_worst = eval_metrics(model, heldout_coefs, heldout_u_true)
        heldout_worst_best = jnp.minimum(heldout_worst_best, heldout_worst)
        epoch_metrics = {
            "epoch": epoch,
            gen_metric_label: float(gen_loss),
            op_metric_label: float(op_loss),
            "heldout_mean_rel_l2": float(heldout_mean),
            "heldout_worst_rel_l2": float(heldout_worst),
            "heldout_worst_best": float(heldout_worst_best),
            "sig_mean": float(model.sampler.sig.mean()),
        }
        metrics = [
            f"epoch={epoch}",
            f"{gen_metric_label}={gen_loss.item():.6f}",
            f"{op_metric_label}={op_loss.item():.6f}",
            f"heldout_mean_rel_l2={heldout_mean.item():.6f}",
            f"heldout_worst_rel_l2={heldout_worst.item():.6f}",
            f"heldout_worst_best={heldout_worst_best.item():.6f}",
        ]
        if divergence_operator is not None:
            epoch_metrics.update(
                {
                    "op_train_rel_l2": float(op_rel_loss),
                    "op_train_pred_div_meanabs": float(op_div_loss),
                }
            )
            metrics.extend(
                [
                    f"op_train_rel_l2={op_rel_loss.item():.6f}",
                    f"op_train_pred_div_meanabs={op_div_loss.item():.6f}",
                ]
            )
            heldout_div_mean, heldout_div_worst = pred_div_metrics(model, heldout_coefs, divergence_operator, divergence_indices)
            heldout_div_worst_best = jnp.minimum(heldout_div_worst_best, heldout_div_worst)
            epoch_metrics.update(
                {
                    "heldout_mean_pred_div_meanabs": float(heldout_div_mean),
                    "heldout_worst_pred_div_meanabs": float(heldout_div_worst),
                    "heldout_worst_pred_div_meanabs_best": float(heldout_div_worst_best),
                }
            )
            metrics.extend(
                [
                    f"heldout_mean_pred_div_meanabs={heldout_div_mean.item():.6f}",
                    f"heldout_worst_pred_div_meanabs={heldout_div_worst.item():.6f}",
                    f"heldout_worst_pred_div_meanabs_best={heldout_div_worst_best.item():.6f}",
                ]
            )
        metrics.append(f"sig_mean={model.sampler.sig.mean().item():.6f}")
        print(" ".join(metrics))
        log_wandb_metrics(wandb_run, epoch_metrics, step=epoch)

        history["epoch"].append(epoch)
        history["gen_loss"].append(float(gen_loss))
        history["op_loss"].append(float(op_loss))
        history["op_rel_loss"].append(float(op_rel_loss))
        history["op_div_loss"].append(float(op_div_loss))
        history["heldout_mean_rel_l2"].append(float(heldout_mean))
        history["heldout_worst_rel_l2"].append(float(heldout_worst))
        history["heldout_worst_rel_l2_best"].append(float(heldout_worst_best))
        if divergence_operator is not None:
            history["heldout_mean_pred_div_meanabs"].append(float(heldout_div_mean))
            history["heldout_worst_pred_div_meanabs"].append(float(heldout_div_worst))
            history["heldout_worst_pred_div_meanabs_best"].append(float(heldout_div_worst_best))
        else:
            history["heldout_mean_pred_div_meanabs"].append(onp.nan)
            history["heldout_worst_pred_div_meanabs"].append(onp.nan)
            history["heldout_worst_pred_div_meanabs_best"].append(onp.nan)
        history["sig_mean"].append(float(model.sampler.sig.mean()))

    metrics_plot_path = (
        Path(args.metrics_plot_path)
        if args.metrics_plot_path
        else Path(f"stokes_kno_minimax_ad_metrics_seed{args.seed}.png")
    )
    config_lines = [
        f"generator_objective={args.generator_objective}",
        f"op_div_weight={args.op_div_weight}",
        f"pred_div_xi={args.pred_div_xi}",
        f"pred_div_region={'all' if args.pred_div_include_boundary else 'interior'}",
        f"boundary_quadrature_rule={args.boundary_quadrature_rule}",
        f"epochs={args.epochs}",
        "generator_updates_per_epoch=1",
        f"op_steps={args.op_steps}",
        f"batch={args.batch}",
        f"lr_gen={args.lr_gen}",
        f"lr_op={args.lr_op}",
        f"seed={args.seed}",
    ]
    metrics_plot_path = save_training_metrics_figure(
        history,
        save_path=metrics_plot_path,
        gen_metric_label=gen_metric_label,
        op_metric_label=op_metric_label,
        config_lines=config_lines,
    )
    print(f"saved_metrics_plot={metrics_plot_path}")
    finish_wandb_run(wandb_run, metrics_plot_path=metrics_plot_path)


if __name__ == "__main__":
    main()
