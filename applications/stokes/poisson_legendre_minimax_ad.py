from __future__ import annotations

import argparse
import contextlib
import logging
import os
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import optax

from local_sampler_utils import build_sampler, sample_noise_batch, sampler_scale_mean
from poisson_legendre_basis import (
    generate_total_degree_multi_indices,
    legendre_vandermonde,
)
from poisson_legendre_operator import (
    PointSetKNOHead,
    PoissonLegendreMinimaxModel,
    ScalarLegendreKNO,
    build_operator_filter,
    rel_l2,
)
from poisson_legendre_solver_ad import PoissonSolutionSolver, build_problem, build_vertex_quadrature
from poisson_transolver_operator import ScalarLegendreTransolver, build_transolver_filter
from poisson_training_metric_plots import save_poisson_training_metrics_figure
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


def build_model(
    *,
    legendre_degree,
    mesh_nx,
    mesh_ny,
    lift_dim,
    depth,
    kernel_jitter,
    operator_type,
    transolver_heads,
    transolver_slices,
    transolver_mlp_ratio,
    sampler_type,
    sampler_components,
    sampler_temperature,
    quiet_solver,
    key,
):
    key_sampler, key_operator = jr.split(key)
    multi_indices = generate_total_degree_multi_indices(legendre_degree, 2)

    with maybe_quiet_solver(quiet_solver):
        problem = build_problem(mesh_nx=mesh_nx, mesh_ny=mesh_ny)

    solver_quad_locs = jnp.asarray(problem.fes[0].get_physical_quad_points()).reshape(-1, 2)
    source_basis_quad = legendre_vandermonde(solver_quad_locs, multi_indices)
    source_basis_quad = source_basis_quad.reshape(problem.fes[0].num_cells, problem.fes[0].num_quads, -1)
    problem.set_source_basis(source_basis_quad)
    solver = PoissonSolutionSolver(problem)

    vertex_locs, vertex_quadrature_w = build_vertex_quadrature(problem.fes[0])
    input_basis = legendre_vandermonde(vertex_locs, multi_indices)
    if operator_type == "kno":
        head = PointSetKNOHead(
            lift_dim=lift_dim,
            depth=depth,
            in_feats=vertex_locs.shape[1] + 1,
            kernel_jitter=kernel_jitter,
            key=key_operator,
        )
        operator = ScalarLegendreKNO(
            head=head,
            input_locs=vertex_locs,
            quadrature_locs=vertex_locs,
            quadrature_w=vertex_quadrature_w,
            input_basis=input_basis,
            output_locs=vertex_locs,
        )
        operator_filter = build_operator_filter(operator)
    elif operator_type == "transolver":
        operator = ScalarLegendreTransolver(
            input_locs=vertex_locs,
            input_basis=input_basis,
            output_locs=vertex_locs,
            hidden_dim=lift_dim,
            depth=depth,
            heads=transolver_heads,
            slice_num=transolver_slices,
            mlp_ratio=transolver_mlp_ratio,
            key=key_operator,
        )
        operator_filter = build_transolver_filter(operator)
    else:
        raise ValueError(f"Unsupported operator_type={operator_type}")

    model = PoissonLegendreMinimaxModel(
        sampler=build_sampler(
            sampler_type=sampler_type,
            dim=input_basis.shape[1],
            key=key_sampler,
            num_components=sampler_components,
            temperature=sampler_temperature,
        ),
        operator=operator,
    )
    return model, solver, multi_indices, solver_quad_locs, operator_filter


@eqx.filter_jit
def sample_coefs_batch(sampler, noise_batch):
    return jax.vmap(sampler.sample_eps)(noise_batch)


@eqx.filter_jit
def mean_batch_loss(model, coefs_batch, u_true_batch):
    losses = jax.vmap(lambda coefs, u_true: rel_l2(u_true, model.predict(coefs)))(coefs_batch, u_true_batch)
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
    rel_errors = jax.vmap(lambda coefs, u_true: rel_l2(u_true, model.predict(coefs)))(coefs_batch, u_true_batch)
    return rel_errors.mean(), rel_errors.max()


@eqx.filter_jit
def fgsm_attack_batch(model, coefs_batch, u_true_batch, epsilon):
    grad_coefs = jax.vmap(jax.grad(lambda coefs, u_true: rel_l2(u_true, model.predict(coefs)), argnums=0))(
        coefs_batch,
        u_true_batch,
    )
    return coefs_batch + epsilon * jnp.sign(grad_coefs)


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
    parser.add_argument("--legendre-degree", type=int, default=3)
    parser.add_argument("--mesh-nx", type=int, default=64)
    parser.add_argument("--mesh-ny", type=int, default=64)
    parser.add_argument("--lift-dim", type=int, default=8)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--kernel-jitter", type=float, default=1e-3)
    parser.add_argument("--operator-type", choices=("kno", "transolver"), default="transolver")
    parser.add_argument("--transolver-heads", type=int, default=4)
    parser.add_argument("--transolver-slices", type=int, default=32)
    parser.add_argument("--transolver-mlp-ratio", type=int, default=1)
    parser.add_argument("--sampler-type", choices=("gaussian", "mog"), default="gaussian")
    parser.add_argument("--sampler-components", type=int, default=4)
    parser.add_argument("--sampler-temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--op-steps", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heldout-batch", type=int, default=100)
    parser.add_argument("--heldout-seed", type=int, default=123)
    parser.add_argument("--eval-fgsm", action="store_true")
    parser.add_argument("--fgsm-epsilon", type=float, default=0.0)
    parser.add_argument("--fgsm-heldout-count", type=int, default=0)
    parser.add_argument("--fgsm-eval-every", type=int, default=1)
    parser.add_argument("--lr-gen", type=float, default=1e-3)
    parser.add_argument("--lr-op", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet-solver", action="store_true")
    parser.add_argument("--freeze-generator", action="store_true")
    parser.add_argument("--metrics-plot-path", type=str, default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="poisson_legendre_minimax")
    parser.add_argument("--wandb-name", type=str, default="")
    args = parser.parse_args()

    master_key = jr.PRNGKey(args.seed)
    key_model, key_train = jr.split(master_key)
    model, solver, multi_indices, solver_quad_locs, operator_filter = build_model(
        legendre_degree=args.legendre_degree,
        mesh_nx=args.mesh_nx,
        mesh_ny=args.mesh_ny,
        lift_dim=args.lift_dim,
        depth=args.depth,
        kernel_jitter=args.kernel_jitter,
        operator_type=args.operator_type,
        transolver_heads=args.transolver_heads,
        transolver_slices=args.transolver_slices,
        transolver_mlp_ratio=args.transolver_mlp_ratio,
        sampler_type=args.sampler_type,
        sampler_components=args.sampler_components,
        sampler_temperature=args.sampler_temperature,
        quiet_solver=args.quiet_solver,
        key=key_model,
    )

    optimizer_sampler = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_gen))
    optimizer_operator = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_op))
    opt_state_sampler = optimizer_sampler.init(eqx.filter(model.sampler, eqx.is_array))
    opt_state_operator = optimizer_operator.init(eqx.filter(model.operator, operator_filter))

    heldout_noise = sample_noise_batch(model.sampler, jr.PRNGKey(args.heldout_seed), args.heldout_batch)
    heldout_coefs = sample_coefs_batch(model.sampler, heldout_noise)
    heldout_u_true = solve_batch(solver, heldout_coefs, quiet_solver=args.quiet_solver)
    fgsm_enabled = args.eval_fgsm and args.fgsm_epsilon > 0.0 and args.fgsm_heldout_count > 0
    fgsm_eval_every = max(1, args.fgsm_eval_every)
    fgsm_count = min(args.fgsm_heldout_count, args.heldout_batch)
    fgsm_clean_coefs = heldout_coefs[:fgsm_count]
    fgsm_clean_u_true = heldout_u_true[:fgsm_count]

    @eqx.filter_jit
    def train_step_operator(model, opt_state, coefs_batch, u_true_batch):
        operator_params, operator_static = eqx.partition(model.operator, operator_filter)

        def loss_fn(current_params):
            current_operator = eqx.combine(current_params, operator_static)
            current_model = eqx.tree_at(lambda current_model: current_model.operator, model, current_operator)
            return mean_batch_loss(current_model, coefs_batch, u_true_batch)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(operator_params)
        updates, opt_state = optimizer_operator.update(grads, opt_state, operator_params)
        operator_params = eqx.apply_updates(operator_params, updates)
        operator = eqx.combine(operator_params, operator_static)
        model = eqx.tree_at(lambda current_model: current_model.operator, model, operator)
        return model, opt_state, loss

    def train_step_sampler(model, opt_state, noise_batch, u_true_batch, solver_vjps):
        sampler_params, sampler_static = eqx.partition(model.sampler, eqx.is_array)

        def sample_from_params(current_params):
            current_sampler = eqx.combine(current_params, sampler_static)
            return jax.vmap(current_sampler.sample_eps)(noise_batch)

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

    heldout_worst_best = jnp.inf
    fgsm_worst_best = jnp.inf
    history = {
        "epoch": [],
        "gen_loss": [],
        "op_loss": [],
        "heldout_mean_rel_l2": [],
        "heldout_worst_rel_l2": [],
        "heldout_worst_rel_l2_best": [],
        "heldout_fgsm_mean_rel_l2": [],
        "heldout_fgsm_worst_rel_l2": [],
        "heldout_fgsm_worst_rel_l2_best": [],
        "sig_mean": [],
    }

    startup_metrics = {
        "legendre_degree": int(args.legendre_degree),
        "basis_size": int(multi_indices.shape[0]),
        "operator_type": args.operator_type,
        "sampler_type": args.sampler_type,
        "sampler_components": int(model.sampler.num_components),
        "fgsm_enabled": fgsm_enabled,
        "fgsm_eval_every": fgsm_eval_every,
        "source_quad_points": int(solver_quad_locs.shape[0]),
        "input_points": int(model.operator.input_locs.shape[0]),
        "solution_nodes": int(model.operator.output_locs.shape[0]),
        "mesh_nx": int(args.mesh_nx),
        "mesh_ny": int(args.mesh_ny),
    }
    print(
        f"basis_size={startup_metrics['basis_size']}, "
        f"source_quad_points={startup_metrics['source_quad_points']}, "
        f"input_points={startup_metrics['input_points']}, "
        f"solution_nodes={startup_metrics['solution_nodes']}, "
        f"operator_type={startup_metrics['operator_type']}, "
        f"sampler_type={startup_metrics['sampler_type']}, "
        f"sampler_components={startup_metrics['sampler_components']}, "
        f"fgsm_enabled={startup_metrics['fgsm_enabled']}, "
        f"fgsm_eval_every={startup_metrics['fgsm_eval_every']}, "
        f"legendre_degree={startup_metrics['legendre_degree']}, "
        f"mesh={startup_metrics['mesh_nx']}x{startup_metrics['mesh_ny']}"
    )
    wandb_run = (
        init_wandb_run(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            static_metrics=startup_metrics,
        )
        if args.wandb
        else None
    )

    for epoch in range(args.epochs):
        gen_loss = jnp.nan
        op_loss = jnp.nan
        key_train, subkey = jr.split(key_train)
        noise_batch = sample_noise_batch(model.sampler, subkey, args.batch)
        coefs_batch = sample_coefs_batch(model.sampler, noise_batch)
        if args.freeze_generator:
            u_true_batch = solve_batch(solver, coefs_batch, quiet_solver=args.quiet_solver)
            solver_vjps = None
        else:
            u_true_batch, solver_vjps = solve_batch_with_vjps(solver, coefs_batch, quiet_solver=args.quiet_solver)
        for _ in range(args.op_steps):
            model, opt_state_operator, op_loss = train_step_operator(model, opt_state_operator, coefs_batch, u_true_batch)

        if args.freeze_generator:
            gen_loss = mean_batch_loss(model, coefs_batch, u_true_batch)
        else:
            model, opt_state_sampler, gen_loss = train_step_sampler(
                model,
                opt_state_sampler,
                noise_batch,
                u_true_batch,
                solver_vjps,
            )

        heldout_mean, heldout_worst = eval_metrics(model, heldout_coefs, heldout_u_true)
        heldout_worst_best = jnp.minimum(heldout_worst_best, heldout_worst)
        fgsm_mean = jnp.nan
        fgsm_worst = jnp.nan
        fgsm_should_eval = fgsm_enabled and (epoch % fgsm_eval_every == 0)
        if fgsm_should_eval:
            fgsm_adv_coefs = fgsm_attack_batch(model, fgsm_clean_coefs, fgsm_clean_u_true, args.fgsm_epsilon)
            fgsm_adv_u_true = solve_batch(solver, fgsm_adv_coefs, quiet_solver=args.quiet_solver)
            fgsm_mean, fgsm_worst = eval_metrics(model, fgsm_adv_coefs, fgsm_adv_u_true)
            fgsm_worst_best = jnp.minimum(fgsm_worst_best, fgsm_worst)
        metrics = [
            f"epoch={epoch}",
            f"gen_rel_l2={gen_loss.item():.6f}",
            f"op_rel_l2={op_loss.item():.6f}",
            f"heldout_mean_rel_l2={heldout_mean.item():.6f}",
            f"heldout_worst_rel_l2={heldout_worst.item():.6f}",
            f"heldout_worst_best={heldout_worst_best.item():.6f}",
            f"sig_mean={sampler_scale_mean(model.sampler).item():.6f}",
        ]
        if fgsm_should_eval:
            metrics.extend(
                [
                    f"heldout_fgsm_mean_rel_l2={fgsm_mean.item():.6f}",
                    f"heldout_fgsm_worst_rel_l2={fgsm_worst.item():.6f}",
                    f"heldout_fgsm_worst_best={fgsm_worst_best.item():.6f}",
                ]
            )
        print(" ".join(metrics))
        wandb_metrics = {
            "epoch": epoch,
            "gen_rel_l2": float(gen_loss),
            "op_rel_l2": float(op_loss),
            "heldout_mean_rel_l2": float(heldout_mean),
            "heldout_worst_rel_l2": float(heldout_worst),
            "heldout_worst_best": float(heldout_worst_best),
            "sig_mean": float(sampler_scale_mean(model.sampler)),
        }
        if fgsm_should_eval:
            wandb_metrics.update(
                {
                    "heldout_fgsm_mean_rel_l2": float(fgsm_mean),
                    "heldout_fgsm_worst_rel_l2": float(fgsm_worst),
                    "heldout_fgsm_worst_best": float(fgsm_worst_best),
                }
            )
        log_wandb_metrics(
            wandb_run,
            wandb_metrics,
            step=epoch,
        )

        history["epoch"].append(epoch)
        history["gen_loss"].append(float(gen_loss))
        history["op_loss"].append(float(op_loss))
        history["heldout_mean_rel_l2"].append(float(heldout_mean))
        history["heldout_worst_rel_l2"].append(float(heldout_worst))
        history["heldout_worst_rel_l2_best"].append(float(heldout_worst_best))
        history["heldout_fgsm_mean_rel_l2"].append(float(fgsm_mean))
        history["heldout_fgsm_worst_rel_l2"].append(float(fgsm_worst))
        history["heldout_fgsm_worst_rel_l2_best"].append(float(fgsm_worst_best))
        history["sig_mean"].append(float(sampler_scale_mean(model.sampler)))

    metrics_plot_path = (
        Path(args.metrics_plot_path)
        if args.metrics_plot_path
        else Path(f"poisson_legendre_minimax_ad_metrics_seed{args.seed}.png")
    )
    config_lines = [
        f"legendre_degree={args.legendre_degree}",
        f"basis_size={multi_indices.shape[0]}",
        f"operator_type={args.operator_type}",
        f"sampler_type={args.sampler_type}",
        f"sampler_components={model.sampler.num_components}",
        f"sampler_temperature={args.sampler_temperature}",
        f"eval_fgsm={fgsm_enabled}",
        f"fgsm_epsilon={args.fgsm_epsilon}",
        f"fgsm_heldout_count={fgsm_count}",
        f"fgsm_eval_every={fgsm_eval_every}",
        f"source_quad_points={solver_quad_locs.shape[0]}",
        f"vertex_points={model.operator.input_locs.shape[0]}",
        f"mesh_nx={args.mesh_nx}",
        f"mesh_ny={args.mesh_ny}",
        f"lift_dim={args.lift_dim}",
        f"depth={args.depth}",
        f"transolver_heads={args.transolver_heads}",
        f"transolver_slices={args.transolver_slices}",
        f"epochs={args.epochs}",
        "generator_updates_per_epoch=1",
        f"op_steps={args.op_steps}",
        f"batch={args.batch}",
        f"lr_gen={args.lr_gen}",
        f"lr_op={args.lr_op}",
        f"seed={args.seed}",
    ]
    metrics_plot_path = save_poisson_training_metrics_figure(
        history,
        save_path=metrics_plot_path,
        config_lines=config_lines,
    )
    print(f"saved_metrics_plot={metrics_plot_path}")
    finish_wandb_run(wandb_run, metrics_plot_path=metrics_plot_path)


if __name__ == "__main__":
    main()
