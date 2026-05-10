import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc



def gaussian_source_value(t, A, t0, sigma):
    """
    Gaussian pulse in time.
    """
    return A * np.exp(-((t - t0) ** 2) / (sigma ** 2))


def source_at_time_single_point(n, source_idx, t, A, t0, sigma):
    """
    Build a single-point source vector at time t.

    Returns a numpy array s of length n such that
    s[source_idx] = Gaussian(t), and all other entries are zero.
    """
    s = np.zeros(n, dtype=float)
    s[source_idx] = gaussian_source_value(t, A, t0, sigma)
    return s



def build_speed_profile(x, medium_type="homogeneous", seed=0):
    """
    Build c(x) on the grid.

    medium_type:
        - "homogeneous"
        - "layered"
        - "random"
    """
    n = len(x)
    c = np.ones(n, dtype=float)

    if medium_type == "homogeneous":
        c[:] = 1.0

    elif medium_type == "layered":
        for i, xi in enumerate(x):
            if 0.0 <= xi < 0.35:
                c[i] = 1.0
            elif 0.35 <= xi < 0.65:
                c[i] = 0.7
            else:
                c[i] = 1.2

    elif medium_type == "random":
        rng = np.random.default_rng(seed)
        n_layers = 40
        edges = np.linspace(0, n - 1, n_layers + 1, dtype=int)

        for j in range(n_layers):
            left = edges[j]
            right = edges[j + 1]
            cval = rng.uniform(0.7, 1.3)
            c[left:right + 1] = cval

    else:
        raise ValueError(f"Unknown medium_type: {medium_type}")

    return c



def build_operator_matrix(c, dx):
    """
    Build PETSc sparse matrix for the operator:
        (Ap)_i = c_i^2 * (p_{i+1} - 2 p_i + p_{i-1}) / dx^2
    on interior rows, with zero boundary rows.

    This matches the project PDE:
        p_tt = c(x)^2 p_xx + s
    """
    n = len(c)
    Nx = n - 1
    inv_dx2 = 1.0 / dx**2

    A = PETSc.Mat().createAIJ([n, n], nnz=3)
    A.setUp()

    # Boundary rows: zero
    A.setValue(0, 0, 0.0)
    A.setValue(Nx, Nx, 0.0)

    # Interior rows
    for i in range(1, Nx):
        ci2 = c[i] ** 2
        A.setValue(i, i - 1,  ci2 * inv_dx2)
        A.setValue(i, i,     -2.0 * ci2 * inv_dx2)
        A.setValue(i, i + 1,  ci2 * inv_dx2)

    A.assemblyBegin()
    A.assemblyEnd()

    return A



def plot_speed_profile(x, c, medium_type, filename):
    plt.figure(figsize=(10, 4))
    plt.plot(x, c)
    plt.xlabel("x")
    plt.ylabel("c(x)")
    plt.title(f"Wave speed profile: {medium_type}")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_snapshots(x, snapshots, snapshot_indices, dt, filename, title):
    plt.figure(figsize=(10, 6))
    for k in snapshot_indices:
        if k in snapshots:
            plt.plot(x, snapshots[k], label=f"t = {k * dt:.3f}")
    plt.xlabel("x")
    plt.ylabel("pressure")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_signal(tgrid, signal, filename, title, ylabel="signal"):
    plt.figure(figsize=(10, 5))
    plt.plot(tgrid, signal)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_overlay_signals(results_dict, filename, title):
    """
    results_dict example:
        {
            "homogeneous": {"tgrid": ..., "receiver_signal": ...},
            "layered": {"tgrid": ..., "receiver_signal": ...},
        }
    """
    plt.figure(figsize=(10, 5))
    for label, result in results_dict.items():
        plt.plot(result["tgrid"], result["receiver_signal"], label=label)
    plt.xlabel("t")
    plt.ylabel("receiver signal")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()



def save_forward_outputs(result):
    """
    Save data and plots for a forward run.

    result should contain:
        prefix, x, tgrid, c, medium_type,
        receiver_signal, receiver_x,
        snapshots, snapshot_indices, dt
    """
    prefix = result["prefix"]

    np.save(f"{prefix}_x.npy", result["x"])
    np.save(f"{prefix}_t.npy", result["tgrid"])
    np.save(f"{prefix}_c.npy", result["c"])
    np.save(f"{prefix}_receiver_signal.npy", result["receiver_signal"])

    plot_speed_profile(
        result["x"],
        result["c"],
        result["medium_type"],
        f"{prefix}_speed.png"
    )

    plot_snapshots(
        result["x"],
        result["snapshots"],
        result["snapshot_indices"],
        result["dt"],
        f"{prefix}_snapshots.png",
        f"Wave snapshots: {result['medium_type']} medium"
    )

    plot_signal(
        result["tgrid"],
        result["receiver_signal"],
        f"{prefix}_receiver_signal.png",
        f"Receiver signal: {result['medium_type']} medium",
        ylabel=f"p(x_r,t), x_r={result['receiver_x']:.3f}"
    )


def save_time_reversal_outputs(result):
    """
    Save data and plots for a time reversal run.

    result should contain:
        prefix, x, tgrid, reversed_signal, observe_signal,
        observe_x, snapshots, snapshot_indices, dt
    """
    prefix = result["prefix"]

    np.save(f"{prefix}_reversed_signal.npy", result["reversed_signal"])
    np.save(f"{prefix}_observe_signal.npy", result["observe_signal"])

    plot_snapshots(
        result["x"],
        result["snapshots"],
        result["snapshot_indices"],
        result["dt"],
        f"{prefix}_snapshots.png",
        "Wave snapshots: time-reversal run"
    )

    plot_signal(
        result["tgrid"],
        result["reversed_signal"],
        f"{prefix}_reemit_signal.png",
        "Re-emitted time-reversed signal",
        ylabel="re-emitted signal"
    )

    plot_signal(
        result["tgrid"],
        result["observe_signal"],
        f"{prefix}_observe_signal.png",
        "Observed signal at original source location",
        ylabel=f"p(x_obs,t), x_obs={result['observe_x']:.3f}"
    )