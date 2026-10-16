import argparse
import tomllib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import qpinn_sweepable as qpinn_std
import qpinn_sweepable_sym as qpinn_sym

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

jax.config.update("jax_enable_x64", True)


def plot_deformation(x_physics, u_pred, grid_dim, skip=5, save_path=None):
    x_deformed = x_physics + u_pred

    x_pts = x_physics[:, 0]
    y_pts = x_physics[:, 1]
    u_x = x_deformed[:, 0]
    u_y = x_deformed[:, 1]

    if x_pts[0] == x_pts[1]:
        X_init = x_pts.reshape(grid_dim, grid_dim).T
        Y_init = y_pts.reshape(grid_dim, grid_dim).T
        X_def = u_x.reshape(grid_dim, grid_dim).T
        Y_def = u_y.reshape(grid_dim, grid_dim).T
    else:
        X_init = x_pts.reshape(grid_dim, grid_dim)
        Y_init = y_pts.reshape(grid_dim, grid_dim)
        X_def = u_x.reshape(grid_dim, grid_dim)
        Y_def = u_y.reshape(grid_dim, grid_dim)

    if skip > 1:
        sub_idx = np.unique(
            np.concatenate([np.arange(0, grid_dim, skip), [grid_dim - 1]])
        )
    else:
        sub_idx = np.arange(grid_dim)

    fig, ax = plt.subplots(figsize=(3.375, 3.0), layout="constrained")

    # Reference grid
    ax.plot(
        X_init[sub_idx, :].T,
        Y_init[sub_idx, :].T,
        "k:",
        alpha=0.35,
        lw=0.6,
        zorder=1,
        rasterized=True,
    )
    ax.plot(
        X_init[:, sub_idx],
        Y_init[:, sub_idx],
        "k:",
        alpha=0.35,
        lw=0.6,
        zorder=1,
        rasterized=True,
    )

    # Deformed grid
    ax.plot(
        X_def[sub_idx, :].T,
        Y_def[sub_idx, :].T,
        "r-",
        alpha=0.7,
        lw=0.7,
        zorder=2,
        rasterized=True,
    )
    ax.plot(
        X_def[:, sub_idx],
        Y_def[:, sub_idx],
        "r-",
        alpha=0.7,
        lw=0.7,
        zorder=2,
        rasterized=True,
    )

    # Outer boundary highlights
    ax.plot(X_def[0, :], Y_def[0, :], "r-", lw=1.0, zorder=3)
    ax.plot(X_def[-1, :], Y_def[-1, :], "r-", lw=1.0, zorder=3)
    ax.plot(X_def[:, 0], Y_def[:, 0], "r-", lw=1.0, zorder=3)
    ax.plot(X_def[:, -1], Y_def[:, -1], "r-", lw=1.0, zorder=3)

    # Subsampled
    X_init_sub = X_init[np.ix_(sub_idx, sub_idx)]
    Y_init_sub = Y_init[np.ix_(sub_idx, sub_idx)]
    X_def_sub = X_def[np.ix_(sub_idx, sub_idx)]
    Y_def_sub = Y_def[np.ix_(sub_idx, sub_idx)]

    ax.scatter(
        X_init_sub,
        Y_init_sub,
        color="grey",
        s=3.0,
        alpha=0.4,
        zorder=1,
        rasterized=True,
    )
    ax.scatter(
        X_def_sub,
        Y_def_sub,
        color="crimson",
        s=3.0,
        alpha=0.8,
        zorder=2,
        rasterized=True,
    )

    ax.set_title("Domain Deformation Grid")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)

    if save_path:
        out_path = (
            save_path
            if save_path.endswith("_deformation.pdf")
            else save_path.rsplit(".", 1)[0] + "_deformation.pdf"
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved deformation plot to {out_path}")

    return fig, ax


def plot_loss_history(loss_history, save_path=None):
    total = loss_history["total"]
    neumann = loss_history["neumann"]
    physics = loss_history["physics"]
    barrier = loss_history["barrier"]

    fig, (ax_total, ax_comp) = plt.subplots(
        1, 2, figsize=(7.0, 2.6), layout="constrained"
    )

    # Total loss
    ax_total.plot(total, color="darkblue", linewidth=1.5, rasterized=True)
    ax_total.set_yscale("symlog", linthresh=1e-4)
    ax_total.set_title("Total Loss History")
    ax_total.set_xlabel("Training Epochs")
    ax_total.set_ylabel(r"$\mathcal{L}$ (Log Scale)")
    ax_total.grid(True, which="both", linestyle=":", alpha=0.5)

    # Loss components
    ax_comp.plot(neumann, color="seagreen", linewidth=1.3, rasterized=True)
    ax_comp.plot(physics, color="darkorange", linewidth=1.3, rasterized=True)
    ax_comp.plot(barrier, color="crimson", linewidth=1.3, rasterized=True)
    ax_comp.set_yscale("symlog", linthresh=1e-4)
    ax_comp.set_title("Loss Components")
    ax_comp.set_xlabel("Training Epochs")
    ax_comp.set_ylabel(r"$\mathcal{L}$ (Log Scale)")
    ax_comp.grid(True, which="both", linestyle=":", alpha=0.5)

    n_epochs = len(total)

    idx_p = int(n_epochs * 0.60)
    ax_comp.annotate(
        r"$\mathcal{L}_{\mathrm{phys}}$",
        xy=(idx_p, physics[idx_p]),
        xytext=(0, 4),
        textcoords="offset points",
        color="darkorange",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    idx_b = int(n_epochs * 0.45)
    ax_comp.annotate(
        r"$\mathcal{L}_{\mathrm{penalty}}$",
        xy=(idx_b, barrier[idx_b]),
        xytext=(0, 2),
        textcoords="offset points",
        color="crimson",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    idx_n = int(n_epochs * 0.65)
    ax_comp.annotate(
        r"$\mathcal{L}_{\mathrm{Neu}}$",
        xy=(idx_n, neumann[idx_n]),
        xytext=(0, 4),
        textcoords="offset points",
        color="seagreen",
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    if save_path:
        out_path = (
            save_path
            if save_path.endswith("_loss.pdf")
            else save_path.rsplit(".", 1)[0] + "_loss.pdf"
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved loss history plot to {out_path}")

    return fig, (ax_total, ax_comp)


def plot_results(x_physics, u_pred, grid_dim, loss_history=None, skip=5, save=None):
    plot_deformation(x_physics, u_pred, grid_dim=grid_dim, skip=skip, save_path=save)
    if loss_history is not None:
        plot_loss_history(loss_history, save_path=save)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize QPINN deformation fields and training loss curves."
    )
    parser.add_argument(
        "-l",
        "--loss-history",
        type=str,
        default=None,
        help="Path to the loss history .npz file",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to the model config .toml file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Path to the serialised Equinox model (.eqx). Optional if plotting from .npz.",
    )
    parser.add_argument(
        "-p",
        "--prediction",
        type=str,
        default=None,
        help="Path to precomputed prediction .npz file (optional).",
    )
    parser.add_argument(
        "--sym",
        action="store_true",
        help="Use sym model definition (from qpinn_sweepable_sym.py).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=5,
        help="Grid subsampling step factor (default: 5)",
    )
    parser.add_argument(
        "-s",
        "--save",
        type=str,
        default=None,
        help="Base path or prefix to save figures (e.g. figs/run1)",
    )
    args = parser.parse_args()

    if args.prediction:
        pred_data = jnp.load(args.prediction)
        x_physics = pred_data["x_physics"]
        u_pred = pred_data["u_pred"]
        grid_size = int(jnp.sqrt(x_physics.shape[0]))
    elif args.config and args.model:
        with open(args.config, "rb") as f:
            cfg = tomllib.load(f)

        grid_size = cfg["domain"]["grid_size"]
        seed = cfg["meta"]["seed"]
        sampling = cfg["domain"]["sampling"]
        mode = cfg["architecture"]["mode"]

        qpinn_mod = (
            qpinn_sym
            if (args.sym or "sym" in str(args.model) or "sym" in str(args.config))
            else qpinn_std
        )

        _x = qpinn_mod.make_1d_grid(grid_size, ax="x", sampling=sampling)
        _y = qpinn_mod.make_1d_grid(grid_size, ax="y", sampling=sampling)
        X, Y = jnp.meshgrid(_x, _y)
        x_physics = jnp.column_stack([X.ravel(), Y.ravel()])

        model_cls = {
            "separate": qpinn_mod.QPINN_aggregate,
            "shared": qpinn_mod.QPINN_shared,
        }[mode]
        skeleton = model_cls(cfg, jr.PRNGKey(seed))
        model = eqx.tree_deserialise_leaves(args.model, skeleton)
        u_pred = jax.vmap(model)(x_physics)
    else:
        raise ValueError()

    loss_history = None
    if args.loss_history:
        loss_npz = jnp.load(args.loss_history)
        loss_history = {
            "total": loss_npz["total"].ravel(),
            "neumann": loss_npz["neumann"].ravel(),
            "physics": loss_npz["physics"].ravel(),
            "barrier": loss_npz["barrier"].ravel(),
        }

    plot_results(
        x_physics,
        u_pred,
        grid_dim=grid_size,
        loss_history=loss_history,
        skip=args.skip,
        save=args.save,
    )


if __name__ == "__main__":
    main()
