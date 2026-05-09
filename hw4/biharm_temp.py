from firedrake import *
import numpy as np


N = 2
levels = 8
Nfine = N * 2**levels

base_mesh = UnitSquareMesh(N, N, quadrilateral=True)
Hierarchy = MeshHierarchy(base_mesh, levels)
mesh = Hierarchy[-1]

V = FunctionSpace(mesh, "Lagrange", 1)
ME = MixedFunctionSpace([V, V], name=["vorticity", "streamfunction"])
Vv = VectorFunctionSpace(mesh, "Lagrange", 1)

omega_t, psi_t = TestFunctions(ME)

u = Function(ME)
u.subfunctions[0].rename("vorticity")
u.subfunctions[1].rename("streamfunction")

omega, psi = split(u)

x, y = SpatialCoordinate(mesh)


A = Constant(0.1)

T = Function(V, name="Temperature")
T.interpolate((1.0 - y) + A * cos(pi * x))


f = Function(V, name="rhs_dTdx")
f.interpolate(T.dx(0))


Fomega = inner(grad(omega_t), grad(omega)) * dx - omega_t * f * dx
Fpsi = inner(grad(psi_t), grad(psi)) * dx - psi_t * omega * dx
F = Fomega + Fpsi

u.subfunctions[0].interpolate(0.0)
u.subfunctions[1].interpolate(0.0)

bcs = [
    DirichletBC(ME.sub(0), 0.0, "on_boundary"),
    DirichletBC(ME.sub(1), 0.0, "on_boundary")
]

pc = "fieldsplit"

params_general = {
    "snes_type": "ksponly",
    "ksp_rtol": 1e-6,
    "ksp_atol": 1e-10,
    "snes_monitor": None,
    "ksp_monitor": None
}

params = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "mg"
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "mg"
    }
}

params.update(params_general)

problem = NonlinearVariationalProblem(F, u, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.solve()

v = Function(Vv, name="Velocity")
v.interpolate(curl(psi))

outfile = VTKFile("result/biharm_temp.pvd")
outfile.write(T, u.subfunctions[0], u.subfunctions[1], v)

print(f"\nSolved problem on {Nfine} x {Nfine} mesh.")
print("Output written to result/biharm_temp.pvd")