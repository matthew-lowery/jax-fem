from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


def _vmap_module(module, x):
    return eqx.filter_vmap(module)(x)


class ResidualMLP(eqx.Module):
    linear_in: eqx.Module
    linear_out: eqx.Module

    def __init__(self, *, in_dim, hidden_dim, out_dim, key):
        key_in, key_out = jr.split(key)
        self.linear_in = eqx.nn.Linear(in_dim, hidden_dim, key=key_in)
        self.linear_out = eqx.nn.Linear(hidden_dim, out_dim, key=key_out)

    def __call__(self, x):
        x = jax.nn.gelu(_vmap_module(self.linear_in, x))
        return _vmap_module(self.linear_out, x)


class SliceAttentionIrregularMesh(eqx.Module):
    in_project_x: eqx.Module
    in_project_fx: eqx.Module
    in_project_slice: eqx.Module
    to_q: eqx.Module
    to_k: eqx.Module
    to_v: eqx.Module
    to_out: eqx.Module
    temperature: jax.Array
    heads: int
    dim_head: int
    scale: float

    def __init__(self, *, hidden_dim, heads, slice_num, key):
        if hidden_dim % heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by heads={heads}")

        key_x, key_fx, key_slice, key_q, key_k, key_v, key_out = jr.split(key, 7)
        inner_dim = hidden_dim
        self.dim_head = hidden_dim // heads
        self.heads = heads
        self.scale = self.dim_head ** -0.5
        self.in_project_x = eqx.nn.Linear(hidden_dim, inner_dim, key=key_x)
        self.in_project_fx = eqx.nn.Linear(hidden_dim, inner_dim, key=key_fx)
        self.in_project_slice = eqx.nn.Linear(self.dim_head, slice_num, key=key_slice)
        self.to_q = eqx.nn.Linear(self.dim_head, self.dim_head, use_bias=False, key=key_q)
        self.to_k = eqx.nn.Linear(self.dim_head, self.dim_head, use_bias=False, key=key_k)
        self.to_v = eqx.nn.Linear(self.dim_head, self.dim_head, use_bias=False, key=key_v)
        self.to_out = eqx.nn.Linear(inner_dim, hidden_dim, key=key_out)
        self.temperature = jnp.full((heads, 1, 1), 0.5)

    def _reshape_heads(self, x):
        x = x.reshape(x.shape[0], self.heads, self.dim_head)
        return jnp.transpose(x, (1, 0, 2))

    def _apply_last_dim(self, module, x):
        return eqx.filter_vmap(eqx.filter_vmap(module))(x)

    def __call__(self, x):
        fx_mid = self._reshape_heads(_vmap_module(self.in_project_fx, x))
        x_mid = self._reshape_heads(_vmap_module(self.in_project_x, x))

        temp = jnp.clip(self.temperature, 0.1, 5.0)
        slice_logits = self._apply_last_dim(self.in_project_slice, x_mid) / temp
        slice_weights = jax.nn.softmax(slice_logits, axis=-1)
        slice_norm = jnp.sum(slice_weights, axis=1)
        slice_token = jnp.einsum("hnc,hng->hgc", fx_mid, slice_weights)
        slice_token = slice_token / (slice_norm[:, :, None] + 1e-5)

        q = self._apply_last_dim(self.to_q, slice_token)
        k = self._apply_last_dim(self.to_k, slice_token)
        v = self._apply_last_dim(self.to_v, slice_token)
        dots = jnp.einsum("hgd,hkd->hgk", q, k) * self.scale
        attn = jax.nn.softmax(dots, axis=-1)
        out_slice = jnp.einsum("hgk,hkd->hgd", attn, v)

        out_x = jnp.einsum("hgc,hng->hnc", out_slice, slice_weights)
        out_x = jnp.transpose(out_x, (1, 0, 2)).reshape(x.shape[0], -1)
        return _vmap_module(self.to_out, out_x)


class TransolverBlock(eqx.Module):
    ln_1: eqx.Module
    ln_2: eqx.Module
    attn: eqx.Module
    mlp: eqx.Module

    def __init__(self, *, hidden_dim, heads, slice_num, mlp_ratio, key):
        key_attn, key_mlp = jr.split(key)
        self.ln_1 = eqx.nn.LayerNorm(hidden_dim)
        self.ln_2 = eqx.nn.LayerNorm(hidden_dim)
        self.attn = SliceAttentionIrregularMesh(
            hidden_dim=hidden_dim,
            heads=heads,
            slice_num=slice_num,
            key=key_attn,
        )
        self.mlp = ResidualMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim * mlp_ratio,
            out_dim=hidden_dim,
            key=key_mlp,
        )

    def __call__(self, fx):
        fx = self.attn(_vmap_module(self.ln_1, fx)) + fx
        fx = self.mlp(_vmap_module(self.ln_2, fx)) + fx
        return fx


class ScalarLegendreTransolver(eqx.Module):
    preprocess: eqx.Module
    blocks: tuple[eqx.Module, ...]
    output_norm: eqx.Module
    output_linear: eqx.Module
    placeholder: jax.Array
    input_locs: jax.Array
    input_basis: jax.Array
    output_locs: jax.Array

    def __init__(
        self,
        *,
        input_locs,
        input_basis,
        output_locs,
        hidden_dim,
        depth,
        heads,
        slice_num,
        mlp_ratio,
        key,
    ):
        key_pre, key_blocks, key_out = jr.split(key, 3)
        self.preprocess = ResidualMLP(in_dim=input_locs.shape[1] + 1, hidden_dim=hidden_dim * 2, out_dim=hidden_dim, key=key_pre)
        self.blocks = tuple(
            TransolverBlock(
                hidden_dim=hidden_dim,
                heads=heads,
                slice_num=slice_num,
                mlp_ratio=mlp_ratio,
                key=current_key,
            )
            for current_key in jr.split(key_blocks, depth)
        )
        self.output_norm = eqx.nn.LayerNorm(hidden_dim)
        self.output_linear = eqx.nn.Linear(hidden_dim, 1, key=key_out)
        self.placeholder = jr.uniform(key_out, (hidden_dim,), minval=-1.0 / hidden_dim, maxval=1.0 / hidden_dim)
        self.input_locs = input_locs
        self.input_basis = input_basis
        self.output_locs = output_locs

    def source_values(self, coefs):
        return self.input_basis @ coefs

    def __call__(self, coefs):
        source_vals = self.source_values(coefs)[:, None]
        fx = jnp.concatenate((self.input_locs, source_vals), axis=-1)
        fx = self.preprocess(fx) + self.placeholder[None, :]
        for block in self.blocks:
            fx = block(fx)
        fx = _vmap_module(self.output_norm, fx)
        return _vmap_module(self.output_linear, fx)


def build_transolver_filter(operator):
    operator_filter = jax.tree.map(eqx.is_array, operator)
    operator_filter = eqx.tree_at(lambda current: current.input_locs, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.input_basis, operator_filter, replace=False)
    operator_filter = eqx.tree_at(lambda current: current.output_locs, operator_filter, replace=False)
    return operator_filter
