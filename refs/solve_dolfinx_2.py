# pyright: reportMissingImports=false
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI

# aerospace-grade Fluorosilicone / Nitrile rubber params
POISSON = 0.45
YOUNGS = 3.5
TRACTION_X = 2.0

GRID_SIZE = 100
N_LOAD_STEPS = 5

MU = YOUNGS / (2.0 * (1.0 + POISSON))
LDA = (POISSON * YOUNGS) / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))

domain = mesh.create_unit_square(
    MPI.COMM_WORLD, GRID_SIZE, GRID_SIZE, mesh.CellType.quadrilateral
)
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
fdim = domain.topology.dim - 1

left_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], 0.0)
)
left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
u_zero = np.zeros(domain.geometry.dim, dtype=default_scalar_type)
bc_left = fem.dirichletbc(u_zero, left_dofs, V)

right_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], 1.0)
)
sorted_facets = np.argsort(right_facets)
facet_tags = mesh.meshtags(
    domain,
    fdim,
    right_facets[sorted_facets],
    np.full_like(right_facets, 1, dtype=np.int32)[sorted_facets],
)

dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 4})
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

u = fem.Function(V)
v = ufl.TestFunction(V)
d = len(u)
I = ufl.variable(ufl.Identity(d))
F = I + ufl.grad(u)
C = F.T * F
Ic = ufl.tr(C)
J = ufl.det(F)
W = (LDA / 2.0) * (ufl.ln(J) ** 2) + (MU / 2.0) * (Ic - d - 2.0 * ufl.ln(J))

T = fem.Constant(domain, default_scalar_type((0.0, 0.0)))
Pi = W * dx - ufl.dot(T, u) * ds(1)
F_form = ufl.derivative(Pi, u, v)
J_form = ufl.derivative(F_form, u, ufl.TrialFunction(V))

problem = NonlinearProblem(
    F_form,
    u,
    bcs=[bc_left],
    J=J_form,
    petsc_options_prefix="neohooke_",
    petsc_options={
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-8,
        "snes_max_it": 50,
        "snes_monitor": None,
    },
)

load_factors = np.linspace(1.0 / N_LOAD_STEPS, 1.0, N_LOAD_STEPS)
for step, factor in enumerate(load_factors, start=1):
    T.value[:] = (TRACTION_X * factor, 0.0)
    _, converged = problem.solve()
    num_iters = problem.solver.getIterationNumber()
    if domain.comm.rank == 0:
        status = "converged" if converged else "FAILED"
        print(
            f"[load step {step}/{N_LOAD_STEPS}] "
            f"factor={factor:.3f} traction_x={TRACTION_X * factor:.4f} "
            f"iters={num_iters} {status} "
            f"u max/min: {u.x.array.max():.6e} / {u.x.array.min():.6e}"
        )
        # print(
        #     f"num_iters={num_iters!r} type={type(num_iters)}, converged={converged!r}"
        # )
    if not converged:
        raise RuntimeError(
            f"Newton solver did not converge at load factor {factor:.3f} "
            f"(step {step}/{N_LOAD_STEPS}) after {num_iters} iterations"
        )

C_inv = ufl.inv(C)
S = MU * (I - C_inv) + LDA * ufl.ln(J) * C_inv  # 2nd Piola-Kirchhoff stress
sigma = (1.0 / J) * F * S * F.T  # Cauchy stress

s11, s22, s12 = sigma[0, 0], sigma[1, 1], sigma[0, 1]

s33 = (LDA * ufl.ln(J)) / J

von_mises_expr = ufl.sqrt(
    0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * s12**2
)

V_dg = fem.functionspace(domain, ("DG", 2))
von_mises_fn = fem.Function(V_dg)
von_mises_fn.interpolate(
    fem.Expression(von_mises_expr, V_dg.element.interpolation_points)
)

from dolfinx import geometry

N_GRID = 100
x_1d = np.linspace(0.0, 1.0, N_GRID)
y_1d = np.linspace(0.0, 1.0, N_GRID)
Xg, Yg = np.meshgrid(x_1d, y_1d, indexing="ij")
grid_points = np.zeros((N_GRID * N_GRID, 3))
grid_points[:, 0] = Xg.ravel()
grid_points[:, 1] = Yg.ravel()


def eval_on_points(domain, fn, points):
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    cells = np.array(
        [colliding_cells.links(i)[0] for i in range(points.shape[0])], dtype=np.int32
    )
    return fn.eval(points, cells)


u_grid_vals = eval_on_points(domain, u, grid_points).reshape(N_GRID, N_GRID, 2)
vm_grid_vals = eval_on_points(domain, von_mises_fn, grid_points).reshape(N_GRID, N_GRID)
u_grid = u_grid_vals[:, :, 0]
v_grid = u_grid_vals[:, :, 1]

coords = domain.geometry.x
disp = u.x.array.reshape(-1, domain.geometry.dim)
# quads
cells = np.asarray(domain.geometry.dofmaps[0]).reshape(-1, 4)

if domain.comm.rank == 0:
    np.savez(
        "test.npz",
        coords=coords,
        u=disp,
        cells=cells,
        x_1d=x_1d,
        y_1d=y_1d,
        u_grid=u_grid,
        v_grid=v_grid,
        von_mises_grid=vm_grid_vals,
    )
