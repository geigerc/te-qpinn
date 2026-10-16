import argparse
import tomllib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

import qpinn_sweepable as qpinn_std
import qpinn_sweepable_sym as qpinn_sym

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

jax.config.update("jax_enable_x64", True)

_ENTANGL_GATES = {
    "cry": lambda theta, wires: qml.CRY(theta, wires=wires),
    "cnot": lambda theta, wires: qml.CNOT(wires=wires),
    "cz": lambda theta, wires: qml.CZ(wires=wires),
    "isingzz": lambda theta, wires: qml.IsingZZ(theta, wires=wires),
}

_PAULI_GATES = {
    "rx": (qml.RX, 0),
    "ry": (qml.RY, 1),
    "rz": (qml.RZ, 2),
}


def von_neumann_ntropy(rho, eps=1e-12):
    p = jnp.clip(jnp.linalg.eigvalsh(rho), eps, 1.0)
    return -jnp.sum(p * jnp.log2(p))


def build_reup_mask(cfg, n_wires=None, unit=None):
    n_wires = n_wires or cfg["qc"]["n_wires"]
    n_layers = cfg["qc"]["n_layers"]
    mapping = cfg["qc"]["reuploading"]["mapping"]
    wire_control = cfg["qc"]["reuploading"]["wire_control"]

    if mapping == "custom":
        if unit == "x" and "custom_mask_x" in cfg["qc"]["reuploading"]:
            return jnp.array(cfg["qc"]["reuploading"]["custom_mask_x"])
        elif unit == "y" and "custom_mask_y" in cfg["qc"]["reuploading"]:
            return jnp.array(cfg["qc"]["reuploading"]["custom_mask_y"])
        return jnp.array(cfg["qc"]["reuploading"]["custom_mask"])

    if mapping == "full":
        return jnp.ones((n_layers, n_wires), dtype=int)
    elif mapping == "alternating":
        return jnp.broadcast_to(
            jnp.where(jnp.arange(n_layers)[:, None] % 2 == 0, 1, 0), (n_layers, n_wires)
        )
    elif mapping == "first":
        return jnp.broadcast_to(
            jnp.where(jnp.arange(n_layers)[:, None] == 0, 1, 0), (n_layers, n_wires)
        )
    elif mapping == "last":
        return jnp.broadcast_to(
            jnp.where(jnp.arange(n_layers)[:, None] == n_layers - 1, 1, 0),
            (n_layers, n_wires),
        )
    elif mapping == "per-wire":
        return jnp.broadcast_to(
            (jnp.arange(n_wires) == wire_control).astype(int), (n_layers, n_wires)
        )


def build_rhomat_qc(cfg, shared=False, unit=None):
    cfg_wires = cfg["qc"]["n_wires"]
    n_wires = 2 * cfg_wires if shared else cfg_wires

    n_layers = cfg["qc"]["n_layers"]
    entangle_layers = set(cfg["qc"]["entangle"]["layers"])
    entangle_topology = cfg["qc"]["entangle"]["topology"]
    entangl_fn = _ENTANGL_GATES[cfg["qc"]["entangle"]["gate"]]
    rotations = [_PAULI_GATES[g] for g in cfg["qc"]["pauli_gates"]]
    reup_mask = build_reup_mask(cfg, n_wires=n_wires, unit=unit).tolist()  # type:ignore

    if entangle_topology == "linear":
        pairs = [(w, w + 1) for w in range(n_wires - 1)]
    elif entangle_topology == "periodic":
        pairs = [(w, (w + 1) % n_wires) for w in range(n_wires)]
    else:
        pairs = [(0, 1), (1, 0)]

    qdev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(qdev, interface="jax")
    def qc(x, theta, basis):
        for i in range(n_layers):
            for j in range(n_wires):
                if reup_mask[i][j]:  # type:ignore
                    qml.RY(basis[j] * x[j % x.shape[0]], wires=j)

            for j in range(n_wires):
                for g, idx in rotations:
                    g(theta[i, j, idx], wires=j)

            if i in entangle_layers:
                for _, (w1, w2) in enumerate(pairs):
                    entangl_fn(theta[i, w1, 3], wires=[w1, w2])

        return qml.density_matrix(wires=range(n_wires // 2))

    return qc


def compute_ntropy_grid(model, cfg, x_e, mode, is_sym=True):
    grid_size = cfg["domain"]["grid_size"]

    if mode == "shared":
        qc_dm = build_rhomat_qc(cfg, shared=True)

        def ntropy_1p(x):
            x_in = jnp.array([x[0], (2.0 * x[1] - 1.0) ** 2]) if is_sym else x

            basis = jnp.concat([model.emb_x(x_in), model.emb_y(x_in)])
            x_rescaled = 2.0 * x_in - 1.0
            dm = qc_dm(x_rescaled, model.theta, basis)
            return von_neumann_ntropy(dm)

        s_map = jax.vmap(ntropy_1p)(x_e)
        return s_map.reshape(grid_size, grid_size)
    else:
        qc_dm_x = build_rhomat_qc(cfg, unit="x")
        qc_dm_y = build_rhomat_qc(cfg, unit="y")

        def ntropy_1p(x):
            x_in = jnp.array([x[0], (2.0 * x[1] - 1.0) ** 2]) if is_sym else x

            x_rescaled = 2.0 * x_in - 1.0

            basis_x = model.QNN_x.emb(x_in)
            dm_x = qc_dm_x(x_rescaled, model.QNN_x.theta, basis_x)
            s_x = von_neumann_ntropy(dm_x)

            basis_y = model.QNN_y.emb(x_in)
            dm_y = qc_dm_y(x_rescaled, model.QNN_y.theta, basis_y)
            s_y = von_neumann_ntropy(dm_y)

            return jnp.array([s_x, s_y])

        s_map = jax.vmap(ntropy_1p)(x_e)
        sx_grid = s_map[:, 0].reshape(grid_size, grid_size)
        sy_grid = s_map[:, 1].reshape(grid_size, grid_size)
        return jnp.stack([sx_grid, sy_grid])


def compute_ntropy_total(s_grid, x_1d, y_1d):
    return jnp.trapezoid(jnp.trapezoid(s_grid, y_1d, axis=0), x_1d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-f", "--filename", type=str, required=True)
    parser.add_argument("-s", "--save", type=str, default=None)
    parser.add_argument(
        "-sym",
        "--sym",
        "--symmetrized",
        action="store_true",
        dest="sym",
        help="use symmetrized model def (from qpinn_sweepable_sym.py)",
    )
    parser.add_argument(
        "-std",
        "--std",
        "--standard",
        action="store_true",
        dest="std",
        help="use unsymmetrized model def (from qpinn_sweepable.py)",
    )
    args = parser.parse_args()

    if args.std:
        is_sym = False
    elif args.sym:
        is_sym = True
    else:
        is_sym = "sym" in str(args.model) or "sym" in str(args.config)

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    mode = cfg["architecture"]["mode"]
    seed = cfg["meta"]["seed"]

    qpinn_mod = qpinn_sym if is_sym else qpinn_std
    model_cls = {
        "separate": qpinn_mod.QPINN_aggregate,
        "shared": qpinn_mod.QPINN_shared,
    }[mode]
    skeleton = model_cls(cfg, jr.PRNGKey(seed))
    model = eqx.tree_deserialise_leaves(args.model, skeleton)

    pred_data = np.load(args.filename)
    x_1d = pred_data["x_1d"]
    y_1d = pred_data["y_1d"]

    X, Y = np.meshgrid(x_1d, y_1d)
    x_e = jnp.column_stack([jnp.array(X.ravel()), jnp.array(Y.ravel())])

    s_grid = compute_ntropy_grid(model, cfg, x_e, mode, is_sym=is_sym)

    if mode == "shared":
        print(jnp.abs(s_grid - s_grid[::-1]).max())
        s_total = compute_ntropy_total(s_grid, x_1d, y_1d)
        print(f"entropy total: {s_total:.6f} bits·units²")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        c0 = axes[0].contourf(
            X,
            Y,
            pred_data["von_mises_grid"].T,
            levels=50,
            cmap="viridis",
            rasterized=True,
        )
        cb0 = fig.colorbar(c0, ax=axes[0])
        axes[0].set_title(r"Von Mises Stress $\sigma_{vm}$")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        c1 = axes[1].contourf(X, Y, s_grid, levels=50, cmap="magma", rasterized=True)
        cb1 = fig.colorbar(c1, ax=axes[1])
        axes[1].set_title(
            r"Inter-Register Entanglement $S_{u_x \leftrightarrow u_y}$ [bits]"
        )
        axes[1].set_xlabel("x")

        for cb in [cb0, cb1]:
            if cb.solids is not None:
                cb.solids.set_edgecolor("face")
                cb.solids.set_rasterized(True)

    else:
        sx_grid, sy_grid = s_grid[0], s_grid[1]

        print(jnp.abs(sx_grid - sx_grid[::-1]).max())
        print(jnp.abs(sy_grid - sy_grid[::-1]).max())

        sx_total = compute_ntropy_total(sx_grid, x_1d, y_1d)
        sy_total = compute_ntropy_total(sy_grid, x_1d, y_1d)
        s_total = sx_total + sy_total
        print(f"total sx: {sx_total:.6f} | sy: {sy_total:.6f} | total: {s_total:.6f}")

        fig, axes = plt.subplots(1, 3, figsize=(17, 5))

        c0 = axes[0].contourf(
            X,
            Y,
            pred_data["von_mises_grid"].T,
            levels=50,
            cmap="viridis",
            rasterized=True,
        )
        cb0 = fig.colorbar(c0, ax=axes[0])
        axes[0].set_title(r"Von Mises Stress $\sigma_{vm}$")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        c1 = axes[1].contourf(X, Y, sx_grid, levels=50, cmap="magma", rasterized=True)
        cb1 = fig.colorbar(c1, ax=axes[1])
        axes[1].set_title("Internal Entanglement $S_x(x, y)$ [bits]")
        axes[1].set_xlabel("x")

        c2 = axes[2].contourf(X, Y, sy_grid, levels=50, cmap="magma", rasterized=True)
        cb2 = fig.colorbar(c2, ax=axes[2])
        axes[2].set_title("Internal Entanglement $S_y(x, y)$ [bits]")
        axes[2].set_xlabel("x")

        for cb in [cb0, cb1, cb2]:
            if cb.solids is not None:
                cb.solids.set_edgecolor("face")
                cb.solids.set_rasterized(True)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"saved fig to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
