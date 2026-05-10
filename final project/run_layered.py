import numpy as np
from petsc4py import PETSc
from common import (
    build_speed_profile,
    build_operator_matrix,
    gaussian_source_value,
    save_forward_outputs,
)


def main():
    # ============================================================
    # Parameters
    # ============================================================
    L = 1.0
    Nx = 400
    dx = L / Nx

    T = 1.2
    xs = 0.2
    xr = 0.8

    source_A = 1.0
    source_t0 = 0.05
    source_sigma = 0.01

    snapshot_times = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    # ============================================================
    # Grid
    # ============================================================
    x = np.linspace(0.0, L, Nx + 1)

    # layered medium
    medium_type = "layered"
    c = build_speed_profile(x, medium_type=medium_type)
    cmax = np.max(c)

    # CFL
    dt = 0.5 * dx / cmax
    Nt = int(np.ceil(T / dt))
    dt = T / Nt
    tgrid = np.linspace(0.0, T, Nt + 1)

    source_idx = int(round(xs / dx))
    receiver_idx = int(round(xr / dx))

    # ============================================================
    # PETSc objects
    # ============================================================
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

    snapshot_indices = sorted(
        set(int(round(ts / dt)) for ts in snapshot_times if ts <= T)
    )
    snapshots = {}

    if 0 in snapshot_indices:
        snapshots[0] = p_prev.getArray().copy()
    if 1 in snapshot_indices:
        snapshots[1] = p_curr.getArray().copy()


    for nstep in range(1, Nt):
        tn = nstep * dt

        # lap_vec = Aop * p_curr
        Aop.mult(p_curr, lap_vec)

        # source at current time
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

        if nstep + 1 in snapshot_indices:
            snapshots[nstep + 1] = arr_next.copy()

        p_prev, p_curr, p_next = p_curr, p_next, p_prev

    result = {
        "prefix": "layered",
        "x": x,
        "tgrid": tgrid,
        "dt": dt,
        "Nx": Nx,
        "T": T,
        "c": c,
        "medium_type": medium_type,
        "receiver_signal": receiver_signal,
        "receiver_x": x[receiver_idx],
        "snapshots": snapshots,
        "snapshot_indices": snapshot_indices,
    }

    save_forward_outputs(result)

    print("Layered run completed.")
    print("Generated:")
    print("  layered_speed.png")
    print("  layered_snapshots.png")
    print("  layered_receiver_signal.png")


if __name__ == "__main__":
    main()