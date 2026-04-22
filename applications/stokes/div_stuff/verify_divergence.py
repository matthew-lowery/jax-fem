from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat, savemat

try:
    from .rbf_fd_divergence import build_rbffd_divergence, divergence_stats
except ImportError:
    from rbf_fd_divergence import build_rbffd_divergence, divergence_stats


HERE = Path(__file__).resolve().parent


def build_point_cloud(nx: int, ny: int) -> jax.Array:
    xs = jnp.linspace(0.0, 1.0, nx)
    ys = jnp.linspace(0.0, 1.0, ny)
    gx, gy = jnp.meshgrid(xs, ys, indexing="xy")
    points = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)

    interior = (
        (points[:, 0] > 0.0)
        & (points[:, 0] < 1.0)
        & (points[:, 1] > 0.0)
        & (points[:, 1] < 1.0)
    )
    jitter = jnp.stack(
        [
            0.11 * jnp.sin(7.0 * points[:, 0] + 3.0 * points[:, 1]),
            0.07 * jnp.cos(5.0 * points[:, 0] + 11.0 * points[:, 1]),
        ],
        axis=-1,
    )
    step = min(1.0 / max(nx - 1, 1), 1.0 / max(ny - 1, 1))
    return points + interior[:, None] * 0.12 * step * jitter


def sample_fields(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    x0 = x[:, 0]
    x1 = x[:, 1]

    u0 = jnp.stack(
        [
            x0**3 + 2.0 * x0 * x1,
            x1**3 - x0 * x1,
        ],
        axis=-1,
    )
    div0 = 3.0 * x0**2 + 2.0 * x1 + 3.0 * x1**2 - x0

    u1 = jnp.stack(
        [
            x0**2 * x1,
            -x0 * x1**2,
        ],
        axis=-1,
    )
    div1 = jnp.zeros_like(x0)

    fields = jnp.stack([u0, u1], axis=0)
    exact_divs = jnp.stack([div0, div1], axis=0)
    return fields, exact_divs


def run_matlab_driver(work_dir: Path, input_path: Path, output_path: Path, xi: int, s_dim: int) -> None:
    command = (
        f"cd('{work_dir.as_posix()}'); "
        f"addpath('{HERE.as_posix()}'); "
        f"matlab_driver_compare('{input_path.as_posix()}', '{output_path.as_posix()}', {xi}, {s_dim});"
    )
    subprocess.run(["matlab", "-batch", command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi", type=int, default=3)
    parser.add_argument("--nx", type=int, default=11)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--run-matlab", action="store_true")
    args = parser.parse_args()

    x = build_point_cloud(args.nx, args.ny)
    fields, exact_divs = sample_fields(x)

    operator = build_rbffd_divergence(x, args.xi)
    divs_jax = operator(fields)
    analytic_error = divs_jax - exact_divs

    summary: dict[str, float | dict[str, float] | int] = {
        "num_points": int(x.shape[0]),
        "stencil_size": operator.stencil_size,
        "jax_vs_exact_max_abs": float(jnp.max(jnp.abs(analytic_error))),
        "jax_vs_exact_rel_l2": float(jnp.linalg.norm(analytic_error) / jnp.linalg.norm(exact_divs)),
        "jax_stats": divergence_stats(divs_jax, operator.boundary_mask),
    }

    if args.run_matlab:
        with tempfile.TemporaryDirectory(dir=HERE) as temp_dir:
            temp_dir_path = Path(temp_dir)
            input_path = temp_dir_path / "verify_input.mat"
            output_path = temp_dir_path / "verify_output.mat"
            payload = {
                "x_grid": np.asarray(x),
                "y_preds_test": np.asarray(fields),
            }
            savemat(input_path, payload)
            run_matlab_driver(temp_dir_path, input_path, output_path, args.xi, x.shape[1])
            matlab_payload = loadmat(output_path)
            matlab_divs = jnp.asarray(matlab_payload["divs"], dtype=jnp.float64)
            matlab_error = divs_jax - matlab_divs
            summary["jax_vs_matlab_max_abs"] = float(jnp.max(jnp.abs(matlab_error)))
            summary["jax_vs_matlab_rel_l2"] = float(jnp.linalg.norm(matlab_error) / jnp.linalg.norm(matlab_divs))
            summary["matlab_stats"] = {
                key: float(np.asarray(matlab_payload[key]).squeeze())
                for key in ("max_abs_div", "max_abs_div_i")
            }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
