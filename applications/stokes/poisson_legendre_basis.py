from __future__ import annotations

from itertools import combinations_with_replacement, permutations, product

import jax
import jax.numpy as jnp
import numpy as onp


@jax.jit
def legendre_poly(n, x):
    p_nm1 = jnp.ones_like(x)
    p_n = x

    def body(k, value):
        p_nm1, p_n = value
        p_np1 = ((2 * k - 1) * x * p_n - (k - 1) * p_nm1) / k
        return p_n, p_np1

    p_n = jax.lax.fori_loop(2, n + 1, body, (p_nm1, p_n))[1]
    return jnp.where(n == 0, p_nm1, jnp.where(n == 1, x, p_n))


def generate_total_degree_multi_indices(degree, dim):
    indices = set()
    for comb in combinations_with_replacement(range(degree + 1), dim):
        if sum(comb) <= degree:
            indices.update(set(permutations(comb)))
    indices = sorted(indices, key=lambda current: (sum(current), current[::-1]))
    return jnp.asarray(indices, dtype=jnp.int32)


def scale_to_legendre_domain(points, lower=0.0, upper=1.0):
    points = jnp.asarray(points)
    return 2.0 * (points - lower) / (upper - lower) - 1.0


def legendre_vandermonde(points, multi_indices):
    scaled_points = scale_to_legendre_domain(points)
    max_degree = int(jnp.max(multi_indices))
    values_per_axis = []

    for axis in range(scaled_points.shape[1]):
        x = scaled_points[:, axis]
        axis_values = [jnp.ones_like(x)]
        if max_degree >= 1:
            axis_values.append(x)
        for degree in range(2, max_degree + 1):
            p_next = ((2 * degree - 1) * x * axis_values[-1] - (degree - 1) * axis_values[-2]) / degree
            axis_values.append(p_next)
        values_per_axis.append(jnp.stack(axis_values, axis=1))

    vandermonde = jnp.ones((scaled_points.shape[0], multi_indices.shape[0]), dtype=scaled_points.dtype)
    for axis, axis_values in enumerate(values_per_axis):
        vandermonde = vandermonde * axis_values[:, multi_indices[:, axis]]
    return vandermonde


def legendre_gauss_tensor_grid(order, dim=2):
    nodes_1d, weights_1d = onp.polynomial.legendre.leggauss(order)
    nodes_1d = 0.5 * (nodes_1d + 1.0)
    weights_1d = 0.5 * weights_1d

    multi_inds = list(product(range(order), repeat=dim))
    points = onp.asarray([[nodes_1d[index] for index in current] for current in multi_inds], dtype=onp.float64)
    weights = onp.asarray(
        [onp.prod([weights_1d[index] for index in current]) for current in multi_inds],
        dtype=onp.float64,
    )
    return jnp.asarray(points), jnp.asarray(weights[:, None])
