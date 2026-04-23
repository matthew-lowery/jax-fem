from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from local_kno_utils import create_lifted_module as clm


def rel_l2(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return jnp.linalg.norm(y_true - y_pred) / (jnp.linalg.norm(y_true) + 1e-12)


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


class PointSetKNOHead(eqx.Module):
    input_kernel: eqx.Module
    output_kernel: eqx.Module
    integration_kernels: list
    proj_kernels: list
    pointwise_layers: list
    lift_kernel: eqx.Module
    lift_dim: int
    depth: int
    kernel_jitter: float

    def __init__(self, *, lift_dim, depth, in_feats, kernel_jitter, key):
        keys = jr.split(key, 2)
        self.integration_kernels = [clm(MaternC2Kernel, lift_dim=lift_dim, key=current) for current in jr.split(keys[0], depth)]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=current) for current in jr.split(keys[1], depth)]

        keys = jr.split(keys[0], 4)
        self.proj_kernels = [
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]),
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]),
            eqx.nn.Linear(lift_dim, 1, key=keys[2]),
        ]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[3])

        keys = jr.split(keys[0], 2)
        self.input_kernel = MaternC2Kernel(key=keys[0])
        self.output_kernel = MaternC2Kernel(key=keys[1])
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

        for layer in range(self.depth - 1):
            f_q_skip = self.pointwise_layers[layer](f_q.T).T
            f_q = eqx.filter_vmap(
                lambda int_kernel, current_f: integration_transform(int_kernel, current_f),
                in_axes=(eqx.if_array(0), 1),
                out_axes=1,
            )(self.integration_kernels[layer], f_q)
            f_q = jax.nn.gelu(f_q_skip + f_q)

        f_q_skip = self.pointwise_layers[-1](f_q.T).T
        f_q = eqx.filter_vmap(
            lambda int_kernel, current_f: integration_transform(int_kernel, current_f),
            in_axes=(eqx.if_array(0), 1),
            out_axes=1,
        )(self.integration_kernels[-1], f_q)
        f_q = f_q_skip + f_q

        f_q = jax.nn.gelu(eqx.filter_vmap(self.proj_kernels[0])(f_q))
        f_q = jax.nn.gelu(eqx.filter_vmap(self.proj_kernels[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_kernels[2])(f_q).reshape(len(q), 1)

        k_qq = self.stabilized_kernel(self.output_kernel, q)
        k_qy = self.output_kernel(q, y_grid)
        return jnp.einsum("mc,qm->qc", f_q, jnp.linalg.solve(k_qq, k_qy).T)


class ScalarLegendreKNO(eqx.Module):
    head: eqx.Module
    input_locs: jax.Array
    quadrature_locs: jax.Array
    quadrature_w: jax.Array
    input_basis: jax.Array
    output_locs: jax.Array

    def source_values(self, coefs):
        return self.input_basis @ coefs

    def __call__(self, coefs):
        source_vals = self.source_values(coefs)[:, None]
        return self.head(
            f_x=source_vals,
            x_grid=self.input_locs,
            y_grid=self.output_locs,
            q=self.quadrature_locs,
            w=self.quadrature_w,
        )


class PoissonLegendreMinimaxModel(eqx.Module):
    sampler: eqx.Module
    operator: eqx.Module

    def predict(self, coefs):
        return self.operator(coefs)


def build_operator_filter(operator):
    operator_filter = jax.tree.map(eqx.is_array, operator)
    operator_filter = eqx.tree_at(lambda current: current.input_locs, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.quadrature_locs, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.quadrature_w, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.input_basis, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.output_locs, operator_filter, replace=False)
    return operator_filter
