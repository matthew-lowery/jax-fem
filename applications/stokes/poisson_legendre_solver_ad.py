from __future__ import annotations

import copy

import equinox as eqx
import jax.numpy as jnp

from jax_fem.generate_mesh import Mesh, rectangle_mesh
from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper


def left(point):
    return jnp.isclose(point[0], 0.0, atol=1e-6)


def right(point):
    return jnp.isclose(point[0], 1.0, atol=1e-6)


def zero_dirichlet_val(point):
    del point
    return 0.0


class LegendreSourcePoisson(Problem):
    source_basis_quad: jax.Array | None

    def custom_init(self):
        self.source_basis_quad = None

    def get_tensor_map(self):
        return lambda x, source: x

    def get_mass_map(self):
        return lambda u, x, source: jnp.array([source])

    def set_source_basis(self, source_basis_quad):
        self.source_basis_quad = jnp.asarray(source_basis_quad)

    def set_params(self, coefs):
        if self.source_basis_quad is None:
            raise ValueError("source_basis_quad must be set before solving")
        source = jnp.einsum("cqm,m->cq", self.source_basis_quad, coefs)
        self.internal_vars = [source]


def build_problem(mesh_nx=64, mesh_ny=64):
    meshio_mesh = rectangle_mesh(mesh_nx, mesh_ny, 1.0, 1.0)
    mesh = Mesh(meshio_mesh.points[:, :2], meshio_mesh.cells_dict["quad"], ele_type="QUAD4")
    dirichlet_bc_info = [
        [left, right],
        [0, 0],
        [zero_dirichlet_val, zero_dirichlet_val],
    ]
    return LegendreSourcePoisson(
        mesh,
        vec=1,
        dim=2,
        ele_type="QUAD4",
        gauss_order=2,
        dirichlet_bc_info=dirichlet_bc_info,
    )


def build_vertex_quadrature(fe):
    local_weights = jnp.sum(fe.shape_vals[None, :, :] * fe.JxW[:, :, None], axis=1)
    global_weights = jnp.zeros((fe.num_total_nodes,), dtype=jnp.asarray(fe.JxW).dtype)
    global_weights = global_weights.at[jnp.asarray(fe.cells)].add(local_weights)
    return jnp.asarray(fe.points), global_weights[:, None]


class PoissonSolutionSolver(eqx.Module):
    problem: LegendreSourcePoisson
    solver_options: dict
    adjoint_solver_options: dict

    def __init__(self, problem, solver_options=None, adjoint_solver_options=None):
        self.problem = problem
        self.solver_options = {"umfpack_solver": {}} if solver_options is None else solver_options
        self.adjoint_solver_options = copy.deepcopy(self.solver_options) if adjoint_solver_options is None else adjoint_solver_options

    def __call__(self, coefs):
        problem = copy.copy(self.problem)
        return ad_wrapper(problem, self.solver_options, self.adjoint_solver_options)(coefs)[0]
