from __future__ import annotations

import argparse
import copy
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp

from example import StokesFlow, transform_cells
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import ad_wrapper


def left(point):
    return np.isclose(point[0], 0.0, atol=1e-5)


def right(point):
    return np.isclose(point[0], 1.0, atol=1e-5)


def zero_dirichlet_val(point):
    del point
    return 0.0


def get_sine_inflow_fn(coefs, modes):
    coefs = np.asarray(coefs)
    modes = np.asarray(modes)

    def inflow(point):
        y = point[1]
        return np.dot(coefs, np.sin(np.pi * modes * y))

    return inflow


class FourierInflowStokes(StokesFlow):
    def custom_init(self, inflow_modes):
        super().custom_init()
        self.inflow_modes = np.asarray(inflow_modes)

    def set_params(self, coefs):
        if coefs.shape != self.inflow_modes.shape:
            raise ValueError(
                f"Expected coefficient shape {self.inflow_modes.shape}, got {coefs.shape}"
            )

        self.fe_u.dirichlet_bc_info[-1][0] = get_sine_inflow_fn(coefs, self.inflow_modes)
        self.fe_u.update_Dirichlet_boundary_conditions(self.fe_u.dirichlet_bc_info)


def build_problem(num_modes, input_dir=None):
    if input_dir is None:
        input_dir = Path(__file__).resolve().parent / "input"
    else:
        input_dir = Path(input_dir)

    ele_type_u = "TRI6"
    points_u = onp.load(input_dir / "numpy" / "points_u.npy")
    cells_u = onp.load(input_dir / "numpy" / "cells_u.npy")
    cells_u = transform_cells(cells_u, points_u, ele_type_u)
    mesh_u = Mesh(points_u, cells_u)

    ele_type_p = "TRI3"
    points_p = onp.load(input_dir / "numpy" / "points_p.npy")
    cells_p = onp.load(input_dir / "numpy" / "cells_p.npy")
    cells_p = transform_cells(cells_p, points_p, ele_type_p)
    mesh_p = Mesh(points_p, cells_p)

    dirichlet_bc_info_u = [
        [right, right],
        [0, 1],
        [zero_dirichlet_val, zero_dirichlet_val],
    ]
    dirichlet_bc_info_p = [
        [left],
        [0],
        [zero_dirichlet_val],
    ]

    problem = FourierInflowStokes(
        [mesh_u, mesh_p],
        vec=[2, 1],
        dim=2,
        ele_type=[ele_type_u, ele_type_p],
        gauss_order=[2, 2],
        dirichlet_bc_info=[dirichlet_bc_info_u, dirichlet_bc_info_p],
        additional_info=(np.arange(1, num_modes + 1),),
    )
    problem.configure_Dirichlet_BC_for_dolphin()
    return problem


class StokesVelocitySolver(eqx.Module):
    problem: FourierInflowStokes
    solver_options: dict
    adjoint_solver_options: dict

    def __init__(
        self,
        problem,
        solver_options=None,
        adjoint_solver_options=None,
    ):
        self.problem = problem
        self.solver_options = {"umfpack_solver": {}} if solver_options is None else solver_options
        if adjoint_solver_options is None:
            self.adjoint_solver_options = copy.deepcopy(self.solver_options)
        else:
            self.adjoint_solver_options = adjoint_solver_options

    def __call__(self, coefs):
        problem = copy.copy(self.problem)
        sol_list = ad_wrapper(problem, self.solver_options, self.adjoint_solver_options)(coefs)
        return sol_list[0]


def build_solver(
    num_modes,
    solver_options=None,
    adjoint_solver_options=None,
    input_dir=None,
):
    problem = build_problem(num_modes=num_modes, input_dir=input_dir)
    return StokesVelocitySolver(
        problem,
        solver_options=solver_options,
        adjoint_solver_options=adjoint_solver_options,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    args = parser.parse_args()

    solver = build_solver(num_modes=args.num_modes)
    coefs = np.linspace(1.0, 0.25, args.num_modes)

    velocity = solver(coefs)
    objective = lambda current_coefs: np.sum(solver(current_coefs) ** 2)
    grad_val = jax.grad(objective)
    grad_val = jax.jit(grad_val)(coefs)

    fd_grad = []
    for i in range(args.num_modes):
        plus = coefs.at[i].add(args.fd_eps)
        minus = coefs.at[i].add(-args.fd_eps)
        fd_grad.append((objective(plus) - objective(minus)) / (2.0 * args.fd_eps))
    fd_grad = np.asarray(fd_grad)

    print(f"velocity.shape = {velocity.shape}")
    print(f"grad.shape = {grad_val.shape}")
    print(f"ad_grad = {grad_val}")
    print(f"fd_grad = {fd_grad}")


if __name__ == "__main__":
    main()
