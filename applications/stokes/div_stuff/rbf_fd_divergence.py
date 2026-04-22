from __future__ import annotations

import math
from functools import partial

import equinox as eqx
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

MATLAB_RBF_EPS = float(jnp.finfo(jnp.float64).eps)


def pairwise_dist(x: jax.Array, y: jax.Array) -> jax.Array:
    x_sq = jnp.sum(x * x, axis=1)[:, None]
    y_sq = jnp.sum(y * y, axis=1)[None, :]
    sq = jnp.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)
    return jnp.sqrt(sq)


def stencil_knn_indices(x: jax.Array, stencil_size: int, neighbor_chunk_size: int | None = None) -> jax.Array:
    num_points = x.shape[0]
    if neighbor_chunk_size is None or num_points <= neighbor_chunk_size:
        distances = pairwise_dist(x, x)
        return jnp.argsort(distances, axis=1)[:, :stencil_size]

    chunk_indices = []
    for start in range(0, num_points, neighbor_chunk_size):
        stop = min(start + neighbor_chunk_size, num_points)
        distances = pairwise_dist(x[start:stop], x)
        _, nearest = jax.lax.top_k(-distances, stencil_size)
        chunk_indices.append(nearest)
    return jnp.concatenate(chunk_indices, axis=0)


def rbffd_params(s_dim: int, xi: int, theta: int = 1) -> tuple[int, int, int, int]:
    ell = xi + theta - 1
    rbf_exp = ell - 1 if ell % 2 == 0 else ell
    rbf_exp = max(rbf_exp, 5)
    rbf_exp = min(rbf_exp, 11)
    poly_m = math.comb(ell + s_dim, s_dim)
    stencil_size = 2 * poly_m + 1
    return ell, poly_m, stencil_size, rbf_exp


def total_degree_indices(s_dim: int, degree: int) -> jax.Array:
    def compositions(total: int, dim: int) -> list[list[int]]:
        if dim == 1:
            return [[total]]

        rows: list[list[int]] = []
        for first in range(total, -1, -1):
            for tail in compositions(total - first, dim - 1):
                rows.append([first, *tail])
        return rows

    rows = [[0] * s_dim]
    for total in range(1, degree + 1):
        rows.extend(compositions(total, s_dim))
    return jnp.asarray(rows, dtype=jnp.int32)


def jacobi_recurrence(num_terms: int, alpha: float = 0.0, beta: float = 0.0) -> tuple[jax.Array, jax.Array]:
    n = jnp.arange(num_terms, dtype=jnp.float64)
    a = (beta**2 - alpha**2) * jnp.ones_like(n)
    b = jnp.ones_like(n)

    a0 = (beta - alpha) / (alpha + beta + 2.0)
    b0 = jnp.exp(
        (alpha + beta + 1.0) * jnp.log(2.0)
        + jax.lax.lgamma(alpha + 1.0)
        + jax.lax.lgamma(beta + 1.0)
        - jax.lax.lgamma(alpha + beta + 2.0)
    )
    a = a.at[0].set(a0)
    b = b.at[0].set(b0)

    if num_terms > 1:
        a1 = (beta**2 - alpha**2) / ((2.0 + alpha + beta) * (4.0 + alpha + beta))
        b1 = 4.0 * (1.0 + alpha) * (1.0 + beta) / ((2.0 + alpha + beta) ** 2 * (3.0 + alpha + beta))
        a = a.at[1].set(a1)
        b = b.at[1].set(b1)

    if num_terms > 2:
        n_tail = n[2:]
        denom_a = (2.0 * n_tail + alpha + beta) * (2.0 * n_tail + alpha + beta + 2.0)
        denom_b = (
            (2.0 * n_tail + alpha + beta) ** 2
            * (2.0 * n_tail + alpha + beta + 1.0)
            * (2.0 * n_tail + alpha + beta - 1.0)
        )
        a_tail = (beta**2 - alpha**2) / denom_a
        b_tail = 4.0 * n_tail * (n_tail + alpha) * (n_tail + beta) * (n_tail + alpha + beta) / denom_b
        a = a.at[2:].set(a_tail)
        b = b.at[2:].set(b_tail)

    return a, b


def poly_eval(
    a: jax.Array,
    b: jax.Array,
    x: jax.Array,
    max_degree: int,
    derivative_order: int = 0,
) -> jax.Array:
    x = jnp.ravel(x)
    num_x = x.shape[0]
    p = jnp.zeros((num_x, max_degree + 1), dtype=x.dtype)

    p = p.at[:, 0].set(jnp.ones((num_x,), dtype=x.dtype) / jnp.sqrt(b[0]))
    if max_degree > 0:
        p = p.at[:, 1].set((x - a[0]) * p[:, 0] / jnp.sqrt(b[1]))

    for degree in range(1, max_degree):
        next_col = ((x - a[degree]) * p[:, degree] - jnp.sqrt(b[degree]) * p[:, degree - 1]) / jnp.sqrt(b[degree + 1])
        p = p.at[:, degree + 1].set(next_col)

    if derivative_order == 0:
        return p

    for deriv in range(1, derivative_order + 1):
        pd = jnp.zeros_like(p)
        for degree in range(deriv, max_degree + 1):
            if degree == deriv:
                base = jnp.exp(jax.lax.lgamma(deriv + 1.0) - 0.5 * jnp.sum(jnp.log(b[: degree + 1])))
                pd = pd.at[:, degree].set(base)
                continue

            next_col = (
                (x - a[degree - 1]) * pd[:, degree - 1]
                - jnp.sqrt(b[degree - 1]) * pd[:, degree - 2]
                + deriv * p[:, degree - 1]
            ) / jnp.sqrt(b[degree])
            pd = pd.at[:, degree].set(next_col)
        p = pd

    return p


def mpoly_eval(
    x: jax.Array,
    alpha: jax.Array,
    max_degree: int,
    axis_max_degrees: tuple[int, ...],
    derivative_axis: int | None = None,
) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float64)
    if x.ndim == 1:
        x = x[None, :]

    num_points, s_dim = x.shape
    a, b = jacobi_recurrence(max_degree + 1)
    values = jnp.ones((num_points, alpha.shape[0]), dtype=jnp.float64) / jnp.sqrt(b[0] ** s_dim)

    for axis in range(s_dim):
        univariate = poly_eval(
            a,
            b,
            x[:, axis],
            axis_max_degrees[axis],
            1 if derivative_axis == axis else 0,
        )
        values = values * univariate[:, alpha[:, axis]] * jnp.sqrt(b[0])

    return values


def build_local_weights(
    stencil_points: jax.Array,
    poly_indices: jax.Array,
    max_degree: int,
    axis_max_degrees: tuple[int, ...],
    rbf_exp: int,
    eps: float,
) -> jax.Array:
    stencil_size, s_dim = stencil_points.shape
    rd = pairwise_dist(stencil_points, stencil_points)
    a = (rd + eps) ** rbf_exp
    scale = jnp.maximum(rd[0, -1], 1e-14)
    centered = (stencil_points - stencil_points[0]) / scale
    v = mpoly_eval(centered, poly_indices, max_degree, axis_max_degrees)
    zeros = jnp.zeros((v.shape[1], v.shape[1]), dtype=stencil_points.dtype)
    lhs = jnp.block([[a, v], [v.T, zeros]])

    drbf = rbf_exp * (rd[0] + eps) ** (rbf_exp - 2)
    drbf_rows = (stencil_points[0] - stencil_points).T * drbf[None, :]

    poly_rows = []
    for axis in range(s_dim):
        poly_rows.append(mpoly_eval(centered[0], poly_indices, max_degree, axis_max_degrees, axis)[0] / scale)
    poly_rows = jnp.stack(poly_rows, axis=0)

    eval_rows = jnp.concatenate([drbf_rows, poly_rows], axis=1)
    dual = jnp.linalg.solve(lhs.T, eval_rows.T)
    return dual[:stencil_size].T


class RBFFDDivergence(eqx.Module):
    x: jax.Array
    stencil_indices: jax.Array
    weights: jax.Array
    boundary_mask: jax.Array
    xi: int = eqx.field(static=True)
    theta: int = eqx.field(static=True)
    ell: int = eqx.field(static=True)
    poly_m: int = eqx.field(static=True)
    stencil_size: int = eqx.field(static=True)
    rbf_exp: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    @eqx.filter_jit
    def __call__(self, fs: jax.Array) -> jax.Array:
        fs = jnp.asarray(fs, dtype=jnp.float64)
        if fs.ndim == 2:
            fs = fs[None, ...]

        stencil_vals = fs[:, self.stencil_indices, :]
        return jnp.einsum("ndm,bnmd->bn", self.weights, stencil_vals)


def build_rbffd_divergence(
    x: jax.Array,
    xi: int,
    *,
    theta: int = 1,
    eps: float = MATLAB_RBF_EPS,
    neighbor_chunk_size: int | None = None,
) -> RBFFDDivergence:
    x = jnp.asarray(x, dtype=jnp.float64)
    s_dim = x.shape[1]
    ell, poly_m, stencil_size, rbf_exp = rbffd_params(s_dim, xi, theta)
    if x.shape[0] < stencil_size:
        raise ValueError(f"need at least {stencil_size} points for xi={xi}, got {x.shape[0]}")

    poly_indices = total_degree_indices(s_dim, ell)
    axis_max_degrees = tuple(int(jnp.max(poly_indices[:, axis])) for axis in range(s_dim))
    stencil_indices = stencil_knn_indices(x, stencil_size, neighbor_chunk_size)
    stencils = x[stencil_indices]
    weight_builder = jax.vmap(
        lambda stencil: build_local_weights(stencil, poly_indices, ell, axis_max_degrees, rbf_exp, eps)
    )
    weights = weight_builder(stencils)

    mins = jnp.min(x, axis=0)
    maxs = jnp.max(x, axis=0)
    boundary_mask = jnp.any((x == mins) | (x == maxs), axis=1)

    return RBFFDDivergence(
        x=x,
        stencil_indices=stencil_indices,
        weights=weights,
        boundary_mask=boundary_mask,
        xi=xi,
        theta=theta,
        ell=ell,
        poly_m=poly_m,
        stencil_size=stencil_size,
        rbf_exp=rbf_exp,
        eps=eps,
    )


def divergence_stats(divs: jax.Array, boundary_mask: jax.Array) -> dict[str, float]:
    divs = jnp.asarray(divs, dtype=jnp.float64)
    interior = divs[:, ~boundary_mask]
    return {
        "max_abs_div": float(jnp.max(jnp.abs(divs))),
        "max_abs_div_i": float(jnp.max(jnp.abs(interior))),
        "median_abs_div": float(jnp.median(jnp.abs(divs))),
        "median_abs_div_i": float(jnp.median(jnp.abs(interior))),
    }
