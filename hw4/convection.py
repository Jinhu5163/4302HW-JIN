from firedrake import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--Ra", type=float, default=1e2)
parser.add_argument("--N", type=int, default=64)
parser.add_argument("--tmax", type=float, default=100000.0)
parser.add_argument("--dt", type=float, default=100.0)
args, unknown = parser.parse_known_args()

Ra_value = args.Ra
N = args.N
dt = args.dt
tmax = args.tmax

Ra = Constant(Ra_value)

os.makedirs("result", exist_ok=True)

mesh = UnitSquareMesh(N, N, quadrilateral=True)

V = FunctionSpace(mesh, "CG", 1)
ME = MixedFunctionSpace([V, V, V], name=["Temperature", "vorticity", "streamfunction"])
Vv = VectorFunctionSpace(mesh, "CG", 1)

u = Function(ME, name="u")
u_old1 = Function(ME, name="u_old1")  # u^n
u_old2 = Function(ME, name="u_old2")  # u^{n-1}

T, omega, psi = split(u)
T_old1, _, _ = split(u_old1)
T_old2, _, _ = split(u_old2)

q, eta, phi = TestFunctions(ME)

x, y = SpatialCoordinate(mesh)


A = Constant(0.5)

u.subfunctions[0].interpolate(
    (1.0 - y)
    + A * cos(pi * x) * sin(pi * y)
)
u.subfunctions[1].interpolate(0.0)
u.subfunctions[2].interpolate(0.0)

u_old1.assign(u)
u_old2.assign(u)

vel = curl(psi)

Tdot_BE = (T - T_old1) / dt

F_T_BE = (
    Tdot_BE * q * dx
    + dot(vel, grad(T)) * q * dx
    + (1.0 / Ra) * inner(grad(T), grad(q)) * dx
)

F_omega = inner(grad(omega), grad(eta)) * dx - T.dx(0) * eta * dx
F_psi = inner(grad(psi), grad(phi)) * dx - omega * phi * dx

F_BE = F_T_BE + F_omega + F_psi

Tdot_BDF2 = (3.0 * T - 4.0 * T_old1 + T_old2) / (2.0 * dt)

F_T_BDF2 = (
    Tdot_BDF2 * q * dx
    + dot(vel, grad(T)) * q * dx
    + (1.0 / Ra) * inner(grad(T), grad(q)) * dx
)

F_BDF2 = F_T_BDF2 + F_omega + F_psi


bcs = [
    DirichletBC(ME.sub(0), 1.0, 3),              # T = 1 bottom
    DirichletBC(ME.sub(0), 0.0, 4),              # T = 0 top
    DirichletBC(ME.sub(1), 0.0, "on_boundary"), # omega = 0
    DirichletBC(ME.sub(2), 0.0, "on_boundary"), # psi = 0
]


params = {
    "snes_type": "newtonls",
    "snes_rtol": 1e-8,
    "snes_atol": 1e-10,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

problem_BE = NonlinearVariationalProblem(F_BE, u, bcs=bcs)
solver_BE = NonlinearVariationalSolver(problem_BE, solver_parameters=params)

problem_BDF2 = NonlinearVariationalProblem(F_BDF2, u, bcs=bcs)
solver_BDF2 = NonlinearVariationalSolver(problem_BDF2, solver_parameters=params)


outfile = VTKFile(f"result/convection_Ra{int(Ra_value)}_N{N}.pvd")

def write_output(t):
    T_out = u.subfunctions[0]
    omega_out = u.subfunctions[1]
    psi_out = u.subfunctions[2]

    v_out = Function(Vv, name="Velocity")
    v_out.interpolate(curl(psi_out))

    outfile.write(T_out, omega_out, psi_out, v_out, time=t)


def compute_nusselt():
    T_out = u.subfunctions[0]

    top_flux = assemble(
        dot(grad(T_out), as_vector([0.0, 1.0])) * ds(4)
    )

    return abs(top_flux)


t = 0.0
step = 0
write_output(t)

t += dt
step += 1
solver_BE.solve()

u_old2.assign(u_old1)
u_old1.assign(u)

Nu = compute_nusselt()
print(f"step={step}, t={t:.4e}, Nu={Nu:.6f}")
write_output(t)

while t < tmax:
    t += dt
    step += 1

    solver_BDF2.solve()

    u_old2.assign(u_old1)
    u_old1.assign(u)

    if step % 10 == 0:
        Nu = compute_nusselt()
        print(f"step={step}, t={t:.4e}, Nu={Nu:.6f}")
        write_output(t)

Nu = compute_nusselt()
write_output(t)
omega_norm = norm(u.subfunctions[1])
psi_norm = norm(u.subfunctions[2])
print(f"omega norm = {omega_norm:.6e}")
print(f"psi norm = {psi_norm:.6e}")

print("\nFinished convection solve")
print(f"Ra = {Ra_value:.3e}")
print(f"N = {N}")
print(f"dt = {dt}")
print(f"tmax = {tmax}")
print(f"Nusselt number Nu = {Nu:.6f}")
print(f"Output written to result/convection_Ra{int(Ra_value)}_N{N}.pvd")