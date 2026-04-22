from __future__ import annotations

import argparse
from functools import partial

import equinox as eqx
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import optax

from fourier_inflow_ad_v2 import build_solver


def create_lifted_module(base_layer, lift_dim, key):
    keys = jr.split(key, lift_dim)
    return eqx.filter_vmap(lambda current_key: base_layer(key=current_key))(keys)


def trapezoid_rule(n):
    x = jnp.linspace(0.0, 1.0, n)[:, None]
    h = x[1, 0] - x[0, 0]
    w = jnp.full((n, 1), h)
    w = w.at[0, 0].set(h / 2.0)
    w = w.at[-1, 0].set(h / 2.0)
    return x, w


def rel_l2(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return jnp.linalg.norm(y_true - y_pred) / (jnp.linalg.norm(y_true) + 1e-12)


class KernelBase:
    def eval_grid(self, x, y):
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        return jax.vmap(jax.vmap(self.eval, (None, 0)), (0, None))(x, y)


class MaternC2Kernel(eqx.Module, KernelBase):
    raw_scale: jax.Array

    def __init__(self, *, key, **kwargs):
        del kwargs
        self.raw_scale = 0.1 * jr.normal(key, ())

    def eval(self, x, y):
        scale = 0.05 + 0.95 * jax.nn.sigmoid(self.raw_scale)
        r = jnp.linalg.norm(x - y)
        r_scaled = r / scale
        sq5 = jnp.sqrt(5.0)
        return (1.0 + sq5 * r_scaled + (5.0 / 3.0) * r_scaled**2) * jnp.exp(-sq5 * r_scaled)


class FreeFunction(eqx.Module):
    vals: jax.Array

    def __init__(self, *, x_size, key, **kwargs):
        del kwargs
        self.vals = 1e-2 * jr.normal(key, (x_size, x_size))

    def __call__(self, x, y):
        del x, y
        return self.vals


class FreeFunctionKernelInterp(eqx.Module):
    vals: jax.Array
    interp_kernel: eqx.Module
    jitter: float

    def __init__(self, *, x_size, interp_kernel, jitter, key, **kwargs):
        del kwargs
        key_vals, key_kernel = jr.split(key)
        self.vals = 1e-2 * jr.normal(key_vals, (x_size, x_size))
        self.interp_kernel = interp_kernel(key=key_kernel)
        self.jitter = jitter

    def __call__(self, x, y):
        k_xx = self.interp_kernel.eval_grid(x, x) + self.jitter * jnp.eye(len(x))
        k_yx = self.interp_kernel.eval_grid(y, x)
        return self.vals @ jnp.linalg.solve(k_xx, k_yx.T)


class QRTransformFactorizedAdd(eqx.Module):
    integration_functions: list
    lift_dim: int
    ndims: int

    def __init__(self, *, integration_function, lift_dim, ndims, key):
        keys = jr.split(key, ndims)
        self.integration_functions = [
            create_lifted_module(integration_function, lift_dim, current_key) for current_key in keys
        ]
        self.lift_dim = lift_dim
        self.ndims = ndims

    def __call__(self, f_q, q_x_1d, q_w_1d, *args, **kwargs):
        del args, kwargs
        weighted_int_funcs = [
            current(q_x_1d, q_x_1d) * q_w_1d.T for current in self.integration_functions
        ]
        if self.ndims != 1:
            raise NotImplementedError("Only ndims=1 is used here.")
        f_q = f_q.reshape(len(q_x_1d), self.lift_dim)
        return jnp.einsum("qc,ckq->kc", f_q, weighted_int_funcs[0])


class QRTransformFactorizedProd(eqx.Module):
    integration_functions: list
    lift_dim: int
    ndims: int

    def __init__(self, *, integration_function, lift_dim, ndims, key):
        keys = jr.split(key, ndims)
        self.integration_functions = [
            create_lifted_module(integration_function, lift_dim, current_key) for current_key in keys
        ]
        self.lift_dim = lift_dim
        self.ndims = ndims

    def __call__(self, f_q, q_x_1d, q_w_1d, eval_locs, *args, **kwargs):
        del args, kwargs
        if self.ndims != 1:
            raise NotImplementedError("Only ndims=1 is used here.")
        weighted_int_func = eqx.filter_vmap(
            lambda integration_function: integration_function(q_x_1d, eval_locs)
        )(self.integration_functions[0]) * q_w_1d[None]
        f_q = f_q.reshape(len(q_x_1d), self.lift_dim)
        return jnp.einsum("qc,cqe->ec", f_q, weighted_int_func)


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


class GPNeuralOperator1D(eqx.Module):
    input_cov_kernel: eqx.Module
    integral_transforms: list
    nonlin_cov_kernels: list
    output_integral_transform: eqx.Module
    lift_layer: eqx.Module
    drop_layer: eqx.Module
    pointwise_layers: list
    q_mu: list
    inducing_locs: jax.Array
    jitter: float
    lift_dim: int
    depth: int

    def __init__(
        self,
        *,
        input_cov_kernel,
        integral_transform,
        nonlin_cov_kernel,
        output_integral_transform,
        num_inducing,
        depth,
        lift_dim,
        in_feats,
        out_feats,
        jitter,
        key,
    ):
        key_input, key_int, key_nonlin, key_out, key_affine, key_gp = jr.split(key, 6)
        self.input_cov_kernel = input_cov_kernel(key=key_input)
        self.integral_transforms = [
            integral_transform(key=current_key) for current_key in jr.split(key_int, depth)
        ]
        self.nonlin_cov_kernels = [
            nonlin_cov_kernel(key=current_key) for current_key in jr.split(key_nonlin, depth - 1)
        ]
        self.output_integral_transform = output_integral_transform(key=key_out)
        key_lift, key_drop, key_pointwise = jr.split(key_affine, 3)
        self.lift_layer = eqx.nn.Linear(in_feats, lift_dim, key=key_lift)
        self.drop_layer = eqx.nn.Linear(lift_dim, out_feats, key=key_drop)
        self.pointwise_layers = [
            eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=current_key)
            for current_key in jr.split(key_pointwise, depth)
        ]
        self.q_mu = [
            1e-3 * jr.normal(current_key, (num_inducing,))
            for current_key in jr.split(key_gp, depth - 1)
        ]
        self.inducing_locs = jnp.linspace(0.0, 1.0, num_inducing)[:, None]
        self.jitter = jitter
        self.lift_dim = lift_dim
        self.depth = depth

    def __call__(self, f_x, x_grid, y_grid, q_x, q_w):
        f_x = jnp.concatenate((f_x, x_grid), axis=-1)
        f_x = eqx.filter_vmap(self.lift_layer)(f_x).reshape(len(x_grid), self.lift_dim)

        k_xx = self.input_cov_kernel.eval_grid(x_grid, x_grid) + self.jitter * jnp.eye(len(x_grid))
        k_xq = self.input_cov_kernel.eval_grid(x_grid, q_x)
        k_qx_k_xx_inv = jnp.linalg.solve(k_xx, k_xq).T
        f_q = jnp.einsum("mc,qm->qc", f_x, k_qx_k_xx_inv)

        for i in range(self.depth - 1):
            f_q = self.pointwise_layers[i](f_q.T).T + self.integral_transforms[i](f_q, q_x, q_w)
            k_bb = self.nonlin_cov_kernels[i].eval_grid(self.inducing_locs, self.inducing_locs)
            k_bb = k_bb + self.jitter * jnp.eye(len(self.inducing_locs))
            coeffs = jnp.linalg.solve(k_bb, self.q_mu[i])
            shape = f_q.shape
            f_q = (
                self.nonlin_cov_kernels[i].eval_grid(f_q.reshape(-1), self.inducing_locs) @ coeffs
            ).reshape(shape)

        f_q = self.pointwise_layers[-1](f_q.T).T + self.integral_transforms[-1](f_q, q_x, q_w)
        f_q = eqx.filter_vmap(self.drop_layer)(f_q)
        return self.output_integral_transform(f_q, q_x, q_w, eval_locs=y_grid)


class StokesMinimaxModel(eqx.Module):
    sampler: eqx.Module
    operator: eqx.Module
    solver: eqx.Module
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

    def rel_l2_from_coefs(self, coefs):
        u_true = self.solver(coefs)
        u_pred = self.predict(coefs)
        return rel_l2(u_true, u_pred)

    def rel_l2_from_eps(self, eps):
        return self.rel_l2_from_coefs(self.sampler.sample_eps(eps))

    def __call__(self, key):
        return self.rel_l2_from_coefs(self.sampler.rvs(key, shape=()))


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
    y_grid = jnp.linspace(0.0, 1.0, solver.num_velocity_nodes)[:, None]

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
    return StokesMinimaxModel(
        sampler=sampler,
        operator=operator,
        solver=solver,
        x_grid=x_grid,
        q_x=q_x,
        q_w=q_w,
        y_grid=y_grid,
    )


def build_filters(model):
    false_tree = jax.tree.map(lambda _: False, model)
    sampler_filter = eqx.tree_at(
        lambda current_model: current_model.sampler,
        false_tree,
        replace=jax.tree.map(eqx.is_array, model.sampler),
    )
    operator_filter = eqx.tree_at(
        lambda current_model: current_model.operator,
        false_tree,
        replace=jax.tree.map(eqx.is_array, model.operator),
    )
    return sampler_filter, operator_filter


def mean_key_loss(model, keys):
    return jax.lax.map(lambda current_key: model(current_key), keys).mean()


def mean_eps_loss(model, eps_batch):
    return jax.lax.map(lambda eps: model.rel_l2_from_eps(eps), eps_batch).mean()


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
    model = build_model(
        num_modes=args.num_modes,
        quadrature_res=args.quadrature_res,
        lift_dim=args.lift_dim,
        depth=args.depth,
        num_inducing=args.num_inducing,
        jitter=args.jitter,
        key=key,
    )
    sampler_filter, operator_filter = build_filters(model)

    optimizer_sampler = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_gen))
    optimizer_operator = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr_op))
    opt_state_sampler = optimizer_sampler.init(eqx.filter(model, sampler_filter))
    opt_state_operator = optimizer_operator.init(eqx.filter(model, operator_filter))

    eval_eps = jr.normal(jr.fold_in(key, 1), (args.eval_batch, args.num_modes))

    @eqx.filter_jit
    def train_step_sampler(model, opt_state, key):
        keys = jr.split(key, args.gen_batch)
        params, static = eqx.partition(model, sampler_filter)

        def loss_fn(current_params):
            current_model = eqx.combine(current_params, static)
            return -mean_key_loss(current_model, keys)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer_sampler.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return eqx.combine(params, static), opt_state, loss

    @eqx.filter_jit
    def train_step_operator(model, opt_state, key):
        keys = jr.split(key, args.op_batch)
        params, static = eqx.partition(model, operator_filter)

        def loss_fn(current_params):
            current_model = eqx.combine(current_params, static)
            return mean_key_loss(current_model, keys)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer_operator.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return eqx.combine(params, static), opt_state, loss

    @eqx.filter_jit
    def eval_model(model, eps_batch):
        return mean_eps_loss(model, eps_batch)

    print(
        f"velocity_nodes={model.solver.num_velocity_nodes}, "
        f"operator_grid={model.y_grid.shape[0]}, "
        f"sampler_dim={args.num_modes}"
    )
    for epoch in range(args.epochs):
        for gen_step in range(args.gen_steps):
            key = jr.fold_in(key, epoch * max(args.gen_steps, 1) + gen_step)
            model, opt_state_sampler, gen_loss = train_step_sampler(model, opt_state_sampler, key)
        for op_step in range(args.op_steps):
            key = jr.fold_in(key, 10_000 + epoch * max(args.op_steps, 1) + op_step)
            model, opt_state_operator, op_loss = train_step_operator(model, opt_state_operator, key)
        eval_loss = eval_model(model, eval_eps)
        print(
            f"epoch={epoch} "
            f"gen_obj={gen_loss.item():.6f} "
            f"op_obj={op_loss.item():.6f} "
            f"eval_rel_l2={eval_loss.item():.6f} "
            f"sig_mean={model.sampler.sig.mean().item():.6f}"
        )


if __name__ == "__main__":
    main()
