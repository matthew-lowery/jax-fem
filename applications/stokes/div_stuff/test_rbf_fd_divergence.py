from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat, savemat

from div_stuff.rbf_fd_divergence import build_rbffd_divergence, divergence_stats
from div_stuff.verify_divergence import HERE, build_point_cloud, run_matlab_driver, sample_fields


def build_tie_free_point_cloud(nx: int, ny: int) -> jax.Array:
    x = build_point_cloud(nx, ny)
    num_points = x.shape[0]
    step = min(1.0 / max(nx - 1, 1), 1.0 / max(ny - 1, 1))
    phase = jnp.arange(num_points, dtype=jnp.float64)
    offset = jnp.stack(
        [
            jnp.sin(jnp.sqrt(2.0) * phase),
            jnp.cos(jnp.sqrt(3.0) * phase),
        ],
        axis=-1,
    )
    return x + 1e-4 * step * offset


def linear_field(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x0 = x[:, 0]
    x1 = x[:, 1]
    u = jnp.stack([x0 + 2.0 * x1, -x0 + 3.0 * x1], axis=-1)
    div = jnp.full_like(x0, 4.0)
    return u, div


def quadratic_fields(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x0 = x[:, 0]
    x1 = x[:, 1]

    u0 = jnp.stack(
        [
            x0**2 + x0 * x1 + 2.0 * x1,
            x1**2 - x0 * x1 + 0.5 * x0,
        ],
        axis=-1,
    )
    div0 = x0 + 3.0 * x1

    u1 = jnp.stack(
        [
            x0**2 - x1**2,
            2.0 * x0 * x1,
        ],
        axis=-1,
    )
    div1 = 4.0 * x0

    return jnp.stack([u0, u1], axis=0), jnp.stack([div0, div1], axis=0)


def quartic_field(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x0 = x[:, 0]
    x1 = x[:, 1]
    u = jnp.stack(
        [
            x0**4 + x0**2 * x1**2 + x1,
            x1**4 - x0 * x1**3 + x0**2,
        ],
        axis=-1,
    )
    div = 4.0 * x0**3 - x0 * x1**2 + 4.0 * x1**3
    return u, div


def smooth_field_pair(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x0 = x[:, 0]
    x1 = x[:, 1]

    u0 = jnp.stack(
        [
            jnp.sin(2.0 * jnp.pi * x0) * jnp.cos(jnp.pi * x1) + x0 * x1,
            jnp.exp(x0 - x1) + x0**2,
        ],
        axis=-1,
    )
    div0 = 2.0 * jnp.pi * jnp.cos(2.0 * jnp.pi * x0) * jnp.cos(jnp.pi * x1) - jnp.exp(x0 - x1)

    u1 = jnp.stack(
        [
            jnp.exp(x0 + x1),
            jnp.sin(x0 * x1) + x0,
        ],
        axis=-1,
    )
    div1 = jnp.exp(x0 + x1) + x0 * jnp.cos(x0 * x1)

    return jnp.stack([u0, u1], axis=0), jnp.stack([div0, div1], axis=0)


class RBFFDDivergenceTests(unittest.TestCase):
    def assertArrayClose(self, actual: jax.Array, expected: jax.Array, atol: float) -> None:
        max_abs = float(jnp.max(jnp.abs(actual - expected)))
        self.assertLessEqual(max_abs, atol, msg=f"max_abs={max_abs}")

    def test_linear_field_exact_for_xi1(self) -> None:
        x = build_point_cloud(7, 6)
        operator = build_rbffd_divergence(x, xi=1)
        field, exact_div = linear_field(x)
        pred = operator(field)[0]
        self.assertArrayClose(pred, exact_div, 1e-12)

    def test_quadratic_fields_exact_for_xi2(self) -> None:
        x = build_point_cloud(9, 8)
        operator = build_rbffd_divergence(x, xi=2)
        fields, exact_divs = quadratic_fields(x)
        pred = operator(fields)
        self.assertArrayClose(pred, exact_divs, 2e-12)

    def test_cubic_fields_exact_for_xi3(self) -> None:
        x = build_point_cloud(11, 10)
        operator = build_rbffd_divergence(x, xi=3)
        fields, exact_divs = sample_fields(x)
        pred = operator(fields)
        self.assertArrayClose(pred, exact_divs, 2e-12)

    def test_quartic_field_exact_for_xi4(self) -> None:
        x = build_point_cloud(12, 12)
        operator = build_rbffd_divergence(x, xi=4)
        field, exact_div = quartic_field(x)
        pred = operator(field)[0]
        self.assertArrayClose(pred, exact_div, 1e-11)

    def test_linearity_on_smooth_fields(self) -> None:
        x = build_point_cloud(10, 9)
        operator = build_rbffd_divergence(x, xi=3)
        fields, _ = smooth_field_pair(x)
        coeff_a = 1.7
        coeff_b = -0.35

        combined = coeff_a * fields[0] + coeff_b * fields[1]
        lhs = operator(combined)[0]
        rhs = coeff_a * operator(fields[0])[0] + coeff_b * operator(fields[1])[0]
        self.assertArrayClose(lhs, rhs, 1e-12)

    def test_chunked_stencil_build_matches_full_build(self) -> None:
        x = build_tie_free_point_cloud(11, 10)
        fields, _ = smooth_field_pair(x)
        operator_full = build_rbffd_divergence(x, xi=3)
        operator_chunked = build_rbffd_divergence(x, xi=3, neighbor_chunk_size=17)
        self.assertArrayClose(operator_full(fields), operator_chunked(fields), 1e-12)

    def test_divergence_stats_match_manual_reduction(self) -> None:
        x = build_point_cloud(11, 10)
        operator = build_rbffd_divergence(x, xi=3)
        fields, _ = sample_fields(x)
        divs = operator(fields)
        stats = divergence_stats(divs, operator.boundary_mask)

        interior = divs[:, ~operator.boundary_mask]
        self.assertAlmostEqual(stats["max_abs_div"], float(jnp.max(jnp.abs(divs))))
        self.assertAlmostEqual(stats["max_abs_div_i"], float(jnp.max(jnp.abs(interior))))
        self.assertAlmostEqual(stats["median_abs_div"], float(jnp.median(jnp.abs(divs))))
        self.assertAlmostEqual(stats["median_abs_div_i"], float(jnp.median(jnp.abs(interior))))

    @unittest.skipUnless(os.environ.get("DIV_STUFF_RUN_MATLAB") == "1", "set DIV_STUFF_RUN_MATLAB=1")
    def test_matches_matlab_on_multiple_cases(self) -> None:
        x0 = build_tie_free_point_cloud(9, 8)
        x1 = build_tie_free_point_cloud(11, 10)
        cases = [
            (2, x0, quadratic_fields(x0)[0]),
            (3, x1, jnp.concatenate([sample_fields(x1)[0], smooth_field_pair(x1)[0]], axis=0)),
        ]

        for xi, x, fields in cases:
            with self.subTest(xi=xi, num_points=int(x.shape[0]), batch=int(fields.shape[0])):
                operator = build_rbffd_divergence(x, xi=xi)
                jax_divs = operator(fields)

                with tempfile.TemporaryDirectory(dir=HERE) as temp_dir:
                    temp_dir_path = Path(temp_dir)
                    input_path = temp_dir_path / "verify_input.mat"
                    output_path = temp_dir_path / "verify_output.mat"
                    savemat(
                        input_path,
                        {
                            "x_grid": np.asarray(x),
                            "y_preds_test": np.asarray(fields),
                        },
                    )
                    run_matlab_driver(temp_dir_path, input_path, output_path, xi, x.shape[1])
                    matlab_divs = jnp.asarray(loadmat(output_path)["divs"], dtype=jnp.float64)

                self.assertArrayClose(jax_divs, matlab_divs, 5e-11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
