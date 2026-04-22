from __future__ import annotations

import argparse

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax.experimental.sparse.linalg import spsolve

from fourier_inflow_ad import build_problem
from fourier_inflow_ad import build_solver as build_reference_solver
from jax_fem.solver import get_A


def build_exact_csr_system(problem):
    zero_sol = problem.unflatten_fn_sol_list(onp.zeros(problem.num_total_dofs_all_vars))
    problem.newton_update(zero_sol)
    A_petsc = get_A(problem)
    indptr, indices, data = A_petsc.getValuesCSR()
    return np.asarray(data), np.asarray(indices), np.asarray(indptr)


def build_inflow_data(problem, num_modes):
    inflow_rows = np.asarray(problem.fe_u.node_inds_list[0] * problem.fe_u.vec + problem.fe_u.vec_inds_list[0])
    inflow_y = np.asarray(problem.fe_u.points[np.asarray(problem.fe_u.node_inds_list[0]), 1])
    modes = np.arange(1, num_modes + 1)
    inflow_sine_basis = np.sin(np.pi * inflow_y[:, None] * modes[None, :])
    return inflow_rows, inflow_sine_basis


class PureJaxStokesVelocitySolver(eqx.Module):
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    inflow_rows: np.ndarray
    inflow_sine_basis: np.ndarray
    num_total_dofs: int
    num_velocity_nodes: int
    velocity_vec_dim: int

    def __call__(self, coefs):
        coefs = np.asarray(coefs)
        rhs = np.zeros(self.num_total_dofs, dtype=self.data.dtype)
        rhs = rhs.at[self.inflow_rows].set(self.inflow_sine_basis @ coefs)
        dofs = spsolve(self.data, self.indices, self.indptr, rhs, tol=1e-10)
        velocity_dofs = self.num_velocity_nodes * self.velocity_vec_dim
        return dofs[:velocity_dofs].reshape(self.num_velocity_nodes, self.velocity_vec_dim)


def build_solver(num_modes):
    problem = build_problem(num_modes=num_modes)
    problem.set_params(np.zeros(num_modes))
    data, indices, indptr = build_exact_csr_system(problem)
    inflow_rows, inflow_sine_basis = build_inflow_data(problem, num_modes)
    return PureJaxStokesVelocitySolver(
        data=data,
        indices=indices,
        indptr=indptr,
        inflow_rows=inflow_rows,
        inflow_sine_basis=inflow_sine_basis,
        num_total_dofs=problem.num_total_dofs_all_vars,
        num_velocity_nodes=problem.fe_u.num_total_nodes,
        velocity_vec_dim=problem.fe_u.vec,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--check-grad", action="store_true")
    args = parser.parse_args()

    solver = build_solver(num_modes=args.num_modes)
    reference_solver = build_reference_solver(num_modes=args.num_modes)
    coefs = np.linspace(1.0, 0.25, args.num_modes)

    jit_solve = eqx.filter_jit(lambda current_coefs: solver(current_coefs))
    velocity = jit_solve(coefs)
    reference_velocity = reference_solver(coefs)

    print(f"velocity.shape = {velocity.shape}")
    print(f"max_abs_diff_vs_reference = {np.max(np.abs(velocity - reference_velocity))}")
    print(f"rel_l2_vs_reference = {np.linalg.norm(velocity - reference_velocity) / np.linalg.norm(reference_velocity)}")

    if args.check_grad:
        objective = lambda current_coefs: np.sum(jit_solve(current_coefs) ** 2)
        grad_val = jax.jit(jax.grad(objective))(coefs)
        print(f"grad.shape = {grad_val.shape}")


if __name__ == "__main__":
    main()
