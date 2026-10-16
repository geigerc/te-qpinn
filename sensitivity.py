import argparse
import tomllib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from mpl_toolkits.axes_grid1 import make_axes_locatable

import qpinn_sweepable as qpinn_std
import qpinn_sweepable_sym as qpinn_sym


def build_qc_gamma(cfg, shared=False, unit=None, qpinn_mod=qpinn_std):
    cfg_wires = cfg["qc"]["n_wires"]
    n_wires = 2 * cfg_wires if shared else cfg_wires

    n_layers = cfg["qc"]["n_layers"]
    diff_method = cfg["qc"]["diff_method"]
    entangle_layers = set(cfg["qc"]["entangle"]["layers"])
    entangle_topology = cfg["qc"]["entangle"]["topology"]
    entangl_fn = qpinn_mod._ENTANGL_GATES[cfg["qc"]["entangle"]["gate"]]
    rotations = [qpinn_mod._PAULI_GATES[g] for g in cfg["qc"]["pauli_gates"]]
    reup_mask = qpinn_mod.build_reup_mask(cfg, n_wires=n_wires, unit=unit).tolist()

    if n_wires < 2:
        pairs = []
    elif entangle_topology == "linear":
        pairs = [(w, w + 1) for w in range(n_wires - 1)]
    elif entangle_topology == "periodic":
        pairs = [(w, (w + 1) % n_wires) for w in range(n_wires)]
    else:
        pairs = [(0, 1), (1, 0)]

    qdev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(qdev, interface="jax", diff_method=diff_method)
    def qc_gamma(x, theta, basis, gamma):
        for i in range(n_layers):
            for j in range(n_wires):
                if reup_mask[i][j]:
                    if n_wires == 1:
                        qml.RY(gamma[i, j] * basis[0] * x[0], wires=0)
                        qml.RZ(gamma[i, j] * basis[1] * x[1], wires=0)
                    else:
                        qml.RY(gamma[i, j] * basis[j] * x[j % x.shape[0]], wires=j)

            for j in range(n_wires):
                for g, idx in rotations:
                    g(theta[i, j, idx], wires=j)

            if i in entangle_layers:
                for _, (w1, w2) in enumerate(pairs):
                    entangl_fn(theta[i, w1, 3], wires=[w1, w2])

        if shared:
            return (
                qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(cfg_wires)])),
                qml.expval(
                    qml.sum(*[qml.PauliZ(i) for i in range(cfg_wires, n_wires)])
                ),
            )

        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))

    return qc_gamma


def compute_reup_saliency(model, x_e, cfg, is_sym=False, qpinn_mod=qpinn_std):
    mode = cfg["architecture"]["mode"]
    n_layers = cfg["qc"]["n_layers"]
    cfg_wires = cfg["qc"]["n_wires"]

    if mode == "separate":
        qc_gamma_x = build_qc_gamma(cfg, shared=False, unit="x", qpinn_mod=qpinn_mod)
        qc_gamma_y = build_qc_gamma(cfg, shared=False, unit="y", qpinn_mod=qpinn_mod)

        def saliency_1p(QPINN_unit, x, qc_fn):
            x_in = jnp.array([x[0], (2.0 * x[1] - 1.0) ** 2]) if is_sym else x
            basis = QPINN_unit.emb(x_in)
            x_rescaled = 2.0 * x_in - 1.0
            gamma_init = jnp.ones((n_layers, cfg_wires))
            grad_gamma = jax.grad(
                lambda g: qc_fn(x_rescaled, QPINN_unit.theta, basis, g)
            )(gamma_init)
            return grad_gamma**2

        sal_x = jnp.mean(
            jax.vmap(lambda x: saliency_1p(model.QNN_x, x, qc_gamma_x))(x_e), axis=0
        )
        sal_y = jnp.mean(
            jax.vmap(lambda x: saliency_1p(model.QNN_y, x, qc_gamma_y))(x_e), axis=0
        )

    elif mode == "shared":
        qc_gamma = build_qc_gamma(cfg, shared=True, qpinn_mod=qpinn_mod)
        total_wires = 2 * cfg_wires

        def saliency_shared_1p(x):
            x_in = jnp.array([x[0], (2.0 * x[1] - 1.0) ** 2]) if is_sym else x
            basis = jnp.concat([model.emb_x(x_in), model.emb_y(x_in)])
            x_rescaled = 2.0 * x_in - 1.0
            gamma_init = jnp.ones((n_layers, total_wires))

            grad_ux = jax.grad(
                lambda g: qc_gamma(x_rescaled, model.theta, basis, g)[0]
            )(gamma_init)
            grad_uy = jax.grad(
                lambda g: qc_gamma(x_rescaled, model.theta, basis, g)[1]
            )(gamma_init)
            return grad_ux**2, grad_uy**2

        sal_x_all, sal_y_all = jax.vmap(saliency_shared_1p)(x_e)
        sal_x = jnp.mean(sal_x_all, axis=0)
        sal_y = jnp.mean(sal_y_all, axis=0)

    return sal_x, sal_y  # type:ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument(
        "-s",
        "--sym",
        action="store_true",
        help="use sym model def (from qpinn_sweepable_sym.py).",
    )
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    mode = cfg["architecture"]["mode"]
    cfg_wires = cfg["qc"]["n_wires"]
    n_layers = cfg["qc"]["n_layers"]

    is_sym = args.sym or "sym" in str(args.model) or "sym" in str(args.config)
    qpinn_mod = qpinn_sym if is_sym else qpinn_std

    seed = cfg["meta"]["seed"]
    model_cls = {
        "separate": qpinn_mod.QPINN_aggregate,
        "shared": qpinn_mod.QPINN_shared,
    }[mode]
    skeleton = model_cls(cfg, jr.PRNGKey(seed))
    model = eqx.tree_deserialise_leaves(args.model, skeleton)

    _x = qpinn_mod.make_1d_grid(50, ax="x", sampling="uniform")
    _y = qpinn_mod.make_1d_grid(50, ax="y", sampling="uniform")
    X, Y = jnp.meshgrid(_x, _y)
    x_eval = jnp.column_stack([X.ravel(), Y.ravel()])

    sal_x, sal_y = compute_reup_saliency(
        model, x_eval, cfg, is_sym=is_sym, qpinn_mod=qpinn_mod
    )

    # plotting
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
        }
    )

    cmap_ux = mcolors.LinearSegmentedColormap.from_list(
        "ux_purp", ["#ffffff", "#f1d4ed", "#992a87", "#4a0640"]
    )
    cmap_uy = mcolors.LinearSegmentedColormap.from_list(
        "uy_teal", ["#ffffff", "#ccedea", "#1b8f88", "#03433e"]
    )

    fig, (ax_x, ax_y) = plt.subplots(
        2, 1, figsize=(3.375, 3.8), gridspec_kw={"hspace": 0.35}
    )

    panels = [
        (ax_x, sal_x, cmap_ux, r"$u_x\ \mathrm{Circuit}$", "#881384"),
        (ax_y, sal_y, cmap_uy, r"$u_y\ \mathrm{Circuit}$", "#0b7a74"),
    ]

    for ax, sal_data, cmap, title, color in panels:
        norm = mcolors.Normalize(vmin=float(sal_data.min()), vmax=float(sal_data.max()))
        data_matrix = sal_data.T
        im = ax.imshow(
            data_matrix, cmap=cmap, norm=norm, aspect="equal", rasterized=True
        )

        ax.set_xticks(np.arange(n_layers))
        ax.set_yticks(np.arange(cfg_wires))
        ax.set_yticklabels([rf"$q = {w}$" for w in range(cfg_wires)])

        ax.set_xticks(np.arange(-0.5, n_layers, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, cfg_wires, 1), minor=True)
        ax.grid(which="minor", color="#333333", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", size=0)

        for w in range(cfg_wires):
            for l in range(n_layers):
                val = float(data_matrix[w, l])
                rel_val = norm(val)
                text_color = "white" if rel_val > 0.55 else "#1a1a1a"
                ax.text(
                    l,
                    w,
                    f"{val:.4f}",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color=text_color,
                    family="serif",
                )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.55)
        cb = fig.colorbar(im, cax=cax)

        vmax = float(sal_data.max())
        exp = int(np.floor(np.log10(vmax))) if vmax > 0 else 0
        scale = 10.0**exp
        cb.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v / scale:.1f}")
        )
        cax.set_title(rf"$\times 10^{{{exp}}}$", fontsize=6.8, pad=4)
        cb.ax.tick_params(labelsize=6.8, color="#333333", length=2.5, pad=2)

        ax.text(
            1.05,
            0.5,
            title,
            transform=ax.transAxes,
            color=color,
            rotation=-90,
            va="center",
            ha="center",
            fontsize=8.5,
            weight="semibold",
        )

    ax_x.set_xticklabels([rf"$L_{l}$" for l in range(n_layers)])
    ax_x.xaxis.tick_top()

    ax_y.set_xticks([])

    fig.text(
        0.5,
        0.04,
        r"$\mathrm{Layer\ index}\ (l)$",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    fig.text(
        0.03,
        0.5,
        r"$\mathrm{Wire\ index}\ (q)$",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=8.5,
    )

    if args.save:
        plt.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"saved fig to {args.save}")

    plt.show()


if __name__ == "__main__":
    main()
