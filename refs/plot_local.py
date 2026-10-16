import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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


def plot_stress(filename, save_path=None):
    data = np.load(filename)
    x_1d = data["x_1d"]
    y_1d = data["y_1d"]
    u_grid = data["u_grid"]
    v_grid = data["v_grid"]
    vm_grid = data["von_mises_grid"]

    X_grid, Y_grid = np.meshgrid(x_1d, y_1d, indexing="ij")

    x = X_grid + u_grid
    y = Y_grid + v_grid

    fig, ax = plt.subplots(figsize=(3.375, 2.1), layout="constrained")

    pcm = ax.pcolormesh(
        x,
        y,
        vm_grid,
        cmap="viridis",
        shading="gouraud",
        vmin=0.0,
        vmax=4.0,
        rasterized=True,
    )

    # subsampled
    skip = 10
    ax.plot(x[::skip, :].T, y[::skip, :].T, "k-", lw=0.3, alpha=0.35)
    ax.plot(x[:, ::skip], y[:, ::skip], "k-", lw=0.3, alpha=0.35)

    ax.plot(x[0, :], y[0, :], "k-", lw=0.8)
    ax.plot(x[-1, :], y[-1, :], "k-", lw=0.8)
    ax.plot(x[:, 0], y[:, 0], "k-", lw=0.8)
    ax.plot(x[:, -1], y[:, -1], "k-", lw=0.8)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(r"Von Mises Stress $\sigma_{\mathrm{vm}}$ [MPa]", fontsize=9.5)

    cbar = fig.colorbar(pcm, ax=ax, shrink=0.68, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_u_components(filename, save_path=None):
    data = np.load(filename)
    x_1d = data["x_1d"]
    y_1d = data["y_1d"]
    u_grid = data["u_grid"]
    v_grid = data["v_grid"]

    X_grid, Y_grid = np.meshgrid(x_1d, y_1d, indexing="ij")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), layout="constrained")

    im1 = ax1.contourf(
        X_grid, Y_grid, u_grid, levels=30, cmap="RdBu_r", rasterized=True
    )
    ax1.set_title(r"Displacement $u_x$", fontsize=9.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$X$")
    ax1.set_ylabel(r"$Y$")
    cbar1 = fig.colorbar(im1, ax=ax1, shrink=0.82, pad=0.03)
    cbar1.ax.tick_params(labelsize=8)

    im2 = ax2.contourf(
        X_grid, Y_grid, v_grid, levels=30, cmap="RdBu_r", rasterized=True
    )
    ax2.set_title(r"Displacement $u_y$", fontsize=9.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$X$")
    cbar2 = fig.colorbar(im2, ax=ax2, shrink=0.82, pad=0.03)
    cbar2.ax.tick_params(labelsize=8)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True)
    parser.add_argument("-s", "--save", type=str, default=None)
    args = parser.parse_args()

    plot_stress(args.filename, save_path=args.save)
    plot_u_components(args.filename)
    plt.show()
