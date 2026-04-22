from __future__ import annotations

import argparse
from functools import partial

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import optax

from fourier_inflow_ad import build_solver
from stokes_gp_neurop_minimax import (
    FreeFunction,
    FreeFunctionKernelInterp,
    GaussianSampler,
    GPNeuralOperator1D,
    MaternC2Kernel,
    QRTransformFactorizedAdd,
    QRTransformFactorizedProd,
    rel_l2,
    trapezoid_rule,
)


class StokesReferenceMinimaxModel(eqx.Module):
    sampler: eqx.Module
    operator: eqx.Module
    x_grid: jax.Array
    q_x: jax.Array
    q_w: jax.Array
    y_grid: jax.Array

    def predict(self, coefs):
        return self.operator(
            f_x=coefs[:, None],
            x_grid=self.x_grid,
            y_grid=self.y_grid,
            q_x=self.q_x,
            q_w=self.q_w,
        )


def build_model(
    *,
    num_modes,
    quadrature_res,
    lift_dim,
    depth,
    num_inducing,
    jitter,
    key,
):
    key_sampler, key_operator = jr.split(key)
    solver = build_solver(num_modes=num_modes)
    x_grid = jnp.linspace(0.0, 1.0, num_modes)[:, None]
    q_x, q_w = trapezoid_rule(quadrature_res)
    y_grid = jnp.linspace(0.0, 1.0, solver.problem.fe_u.num_total_nodes)[:, None]

    input_cov_kernel = partial(MaternC2Kernel)
    nonlin_cov_kernel = partial(MaternC2Kernel)
    integration_function = partial(FreeFunction, x_size=quadrature_res)
    output_integration_function = partial(
        FreeFunctionKernelInterp,
        x_size=quadrature_res,
        interp_kernel=nonlin_cov_kernel,
        jitter=jitter,
    )

    sampler = GaussianSampler(dim=num_modes, key=key_sampler)
    operator = GPNeuralOperator1D(
        input_cov_kernel=input_cov_kernel,
        integral_transform=partial(
            QRTransformFactorizedAdd,
            integration_function=integration_function,
            lift_dim=lift_dim,
            ndims=1,
        ),
        nonlin_cov_kernel=nonlin_cov_kernel,
        output_integral_transform=partial(
            QRTransformFactorizedProd,
            integration_function=output_integration_function,
            lift_dim=2,
            ndims=1,
        ),
        num_inducing=num_inducing,
        depth=depth,
        lift_dim=lift_dim,
        in_feats=2,
        out_feats=2,
        jitter=jitter,
        key=key_operator,
    )
    return (
        StokesReferenceMinimaxModel(
            sampler=sampler,
            operator=operator,
            x_grid=x_grid,
            q_x=q_x,
            q_w=q_w,
            y_grid=y_grid,
        ),
        solver,
    )


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

        loss, (grad_coefs, grad_u_true) = jax.value_and_grad(
            loss_fn,
            argnums=(0, 1),
        )(coefs, u_true)
        return loss, grad_coefs, grad_u_true

    losses, grad_coefs, grad_u_true = jax.vmap(one_sample)(coefs_batch, u_true_batch)
    return losses.mean(), grad_coefs, grad_u_true


def solve_batch(solver, coefs_batch):
    return jnp.stack([solver(coefs) for coefs in coefs_batch])


def solve_batch_with_vjps(solver, coefs_batch):
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
    parser.add_argument("--quadrature-res", type=int, default=128)
    parser.add_argument("--lift-dim", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-inducing", type=int, default=32)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--gen-steps", type=int, default=1)
    parser.add_argument("--op-steps", type=int, default=1)
    parser.add_argument("--gen-batch", type=int, default=2)
    parser.add_argument("--op-batch", type=int, default=2)
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--lr-gen", type=float, default=3e-4)
    parser.add_argument("--lr-op", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    key = jr.PRNGKey(args.seed)
    model, solver = build_model(
        num_modes=args.num_modes,
        quadrature_res=args.quadrature_res,
        lift_dim=args.lift_dim,
        depth=args.depth,
        num_inducing=args.num_inducing,
        jitter=args.jitter,
        key=key,
    )

    optimizer_sampler = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_gen))
    optimizer_operator = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_op))
    opt_state_sampler = optimizer_sampler.init(eqx.filter(model.sampler, eqx.is_array))
    opt_state_operator = optimizer_operator.init(eqx.filter(model.operator, eqx.is_array))

    @eqx.filter_jit
    def train_step_operator(model, opt_state, coefs_batch, u_true_batch):
        operator_params, operator_static = eqx.partition(model.operator, eqx.is_array)

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

    @eqx.filter_jit
    def eval_model(model, coefs_batch, u_true_batch):
        return mean_batch_loss(model, coefs_batch, u_true_batch)

    def train_step_sampler(model, opt_state, eps_batch):
        sampler_params, sampler_static = eqx.partition(model.sampler, eqx.is_array)

        def sample_from_params(current_params):
            current_sampler = eqx.combine(current_params, sampler_static)
            return jax.vmap(current_sampler.sample_eps)(eps_batch)

        coefs_batch, sampler_vjp = jax.vjp(sample_from_params, sampler_params)
        u_true_batch, solver_vjps = solve_batch_with_vjps(solver, coefs_batch)
        loss, grad_coefs_op, grad_u_true = loss_input_partials(model, coefs_batch, u_true_batch)

        # Exact sampler gradient: operator path plus PDE adjoint path.
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

    eval_eps = jr.normal(jr.fold_in(key, 1), (args.eval_batch, args.num_modes))

    print(
        f"velocity_nodes={solver.problem.fe_u.num_total_nodes}, "
        f"operator_grid={model.y_grid.shape[0]}, "
        f"sampler_dim={args.num_modes}"
    )
    for epoch in range(args.epochs):
        gen_loss = jnp.nan
        op_loss = jnp.nan

        for _ in range(args.gen_steps):
            key, subkey = jr.split(key)
            eps_batch = jr.normal(subkey, (args.gen_batch, args.num_modes))
            model, opt_state_sampler, gen_loss = train_step_sampler(model, opt_state_sampler, eps_batch)

        for _ in range(args.op_steps):
            key, subkey = jr.split(key)
            eps_batch = jr.normal(subkey, (args.op_batch, args.num_modes))
            coefs_batch = sample_coefs_batch(model.sampler, eps_batch)
            u_true_batch = solve_batch(solver, coefs_batch)
            model, opt_state_operator, op_loss = train_step_operator(
                model,
                opt_state_operator,
                coefs_batch,
                u_true_batch,
            )

        eval_coefs = sample_coefs_batch(model.sampler, eval_eps)
        eval_u_true = solve_batch(solver, eval_coefs)
        eval_loss = eval_model(model, eval_coefs, eval_u_true)
        print(
            f"epoch={epoch} "
            f"gen_rel_l2={gen_loss.item():.6f} "
            f"op_rel_l2={op_loss.item():.6f} "
            f"eval_rel_l2={eval_loss.item():.6f} "
            f"sig_mean={model.sampler.sig.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
