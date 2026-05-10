import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
from common import (
    build_speed_profile,
    build_operator_matrix,
    gaussian_source_value,
)


def run_single_homogeneous_case(Nx):
  
    L = 1.0
    dx = L / Nx

    T = 1.2
    xs = 0.2
    xr = 0.8

    source_A = 1.0
    source_t0 = 0.05
    source_sigma = 0.01

    x = np.linspace(0.0, L, Nx + 1)

    medium_type = "homogeneous"
    c = build_speed_profile(x, medium_type=medium_type)
    cmax = np.max(c)

    # CFL
    dt = 0.5 * dx / cmax
    Nt = int(np.ceil(T / dt))
    dt = T / Nt
    tgrid = np.linspace(0.0, T, Nt + 1)

    source_idx = int(round(xs / dx))
    receiver_idx = int(round(xr / dx))
    n = Nx + 1
    p_prev = PETSc.Vec().createSeq(n)
    p_curr = PETSc.Vec().createSeq(n)
    p_next = PETSc.Vec().createSeq(n)
    lap_vec = PETSc.Vec().createSeq(n)
    source_vec = PETSc.Vec().createSeq(n)

    Aop = build_operator_matrix(c, dx)

    p_prev.set(0.0)

    p_curr.set(0.0)

    source_vec.set(0.0)
    source_vec.setValue(
        source_idx,
        gaussian_source_value(0.0, source_A, source_t0, source_sigma)
    )
    source_vec.assemblyBegin()
    source_vec.assemblyEnd()

    p_curr.axpy(0.5 * dt**2, source_vec)

    arr_curr = p_curr.getArray()
    arr_curr[0] = 0.0
    arr_curr[-1] = 0.0

    receiver_signal = np.zeros(Nt + 1, dtype=float)
    receiver_signal[0] = 0.0
    receiver_signal[1] = p_curr.getValue(receiver_idx)

    for nstep in range(1, Nt):
        tn = nstep * dt


        Aop.mult(p_curr, lap_vec)

        source_vec.set(0.0)
        source_vec.setValue(
            source_idx,
            gaussian_source_value(tn, source_A, source_t0, source_sigma)
        )
        source_vec.assemblyBegin()
        source_vec.assemblyEnd()

        p_curr.copy(p_next)
        p_next.scale(2.0)
        p_next.axpy(-1.0, p_prev)
        p_next.axpy(dt**2, lap_vec)
        p_next.axpy(dt**2, source_vec)

        arr_next = p_next.getArray()
        arr_next[0] = 0.0
        arr_next[-1] = 0.0

        receiver_signal[nstep + 1] = p_next.getValue(receiver_idx)

        p_prev, p_curr, p_next = p_curr, p_next, p_prev

    return {
        "Nx": Nx,
        "x": x,
        "tgrid": tgrid,
        "receiver_signal": receiver_signal,
        "receiver_x": x[receiver_idx],
        "dt": dt,
    }


def main():
    Nx_list = [200, 400, 800]
    results = {}

    for Nx in Nx_list:
        print(f"Running homogeneous grid sensitivity case: Nx = {Nx}")
        results[Nx] = run_single_homogeneous_case(Nx)

    plt.figure(figsize=(10, 5))
    for Nx in Nx_list:
        plt.plot(
            results[Nx]["tgrid"],
            results[Nx]["receiver_signal"],
            label=f"Nx = {Nx}"
        )
    plt.xlabel("t")
    plt.ylabel("receiver signal")
    plt.title("Grid sensitivity in the homogeneous medium")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grid_sensitivity_homogeneous.png", dpi=200)
    plt.close()

    print("Grid sensitivity run completed.")
    print("Generated:")
    print("  grid_sensitivity_homogeneous.png")


if __name__ == "__main__":
    main()