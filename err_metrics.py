import argparse

import matplotlib.pyplot as plt
import numpy as np


def rel_errs_continuous(pred, ref, x_1d, y_1d):
    err = pred - ref
    ref_l2 = np.sqrt(np.trapezoid(np.trapezoid(ref**2, y_1d, axis=1), x_1d, axis=0))
    err_l2 = np.sqrt(np.trapezoid(np.trapezoid(err**2, y_1d, axis=1), x_1d, axis=0))

    rel_l2 = err_l2 / ref_l2 if ref_l2 > 0 else np.inf
    rel_linf = np.max(np.abs(err)) / np.max(np.abs(ref))
    return rel_l2, rel_linf, err


def rel_errs(pred, ref):
    err = pred - ref
    ref_flat = ref.ravel()
    err_flat = err.ravel()

    l2_denom = np.sqrt(np.sum(ref_flat**2))
    linf_denom = np.max(np.abs(ref_flat))

    rel_l2 = np.sqrt(np.sum(err_flat**2)) / l2_denom if l2_denom > 0 else np.inf
    rel_linf = np.max(np.abs(err_flat)) / linf_denom if linf_denom > 0 else np.inf

    return rel_l2, rel_linf, err


def errawr(pred_filename, ref_filename):
    pred_data = np.load(pred_filename)
    ref_data = np.load(ref_filename)

    x_1d = ref_data["x_1d"]
    y_1d = ref_data["y_1d"]

    u_rel_l2, u_rel_linf, u_err = rel_errs_continuous(
        pred_data["u_grid"], ref_data["u_grid"], x_1d, y_1d
    )
    v_rel_l2, v_rel_linf, v_err = rel_errs_continuous(
        pred_data["v_grid"], ref_data["v_grid"], x_1d, y_1d
    )
    vm_rel_l2, vm_rel_linf, vm_err = rel_errs_continuous(
        pred_data["von_mises_grid"], ref_data["von_mises_grid"], x_1d, y_1d
    )

    print(f"\n{pred_filename}")
    print(f"u_x         rel_L2 = {u_rel_l2:.4e}   rel_Linf = {u_rel_linf:.4e}")
    print(f"u_y         rel_L2 = {v_rel_l2:.4e}   rel_Linf = {v_rel_linf:.4e}")
    print(f"von_mises   rel_L2 = {vm_rel_l2:.4e}   rel_Linf = {vm_rel_linf:.4e}")

    return {
        "u": (u_rel_l2, u_rel_linf, u_err),
        "v": (v_rel_l2, v_rel_linf, v_err),
        "vm": (vm_rel_l2, vm_rel_linf, vm_err),
    }


def plot_errs(pred_filename, ref_filename, plot=False):
    errs = errawr(pred_filename, ref_filename)

    if not plot:
        return

    data = np.load(ref_filename)
    x_1d = data["x_1d"]
    y_1d = data["y_1d"]
    u_grid = data["u_grid"]
    v_grid = data["v_grid"]

    X_grid, Y_grid = np.meshgrid(x_1d, y_1d, indexing="ij")

    x = X_grid + u_grid
    y = Y_grid + v_grid

    titles = {
        "u": r"$u_x$ AbsErr",
        "v": r"$u_y$ AbsErr",
        "vm": r"von Mises Stress AbsErr",
    }

    cbar_labels = {
        "u": r"$|u_x^{\mathrm{pred}} - u_x^{\mathrm{ref}}|$",
        "v": r"$|u_y^{\mathrm{pred}} - u_y^{\mathrm{ref}}|$",
        "vm": r"$|\sigma_{\mathrm{vm}}^{\mathrm{pred}} - \sigma_{\mathrm{vm}}^{\mathrm{ref}}|$ [MPa]",
    }

    shared_cmap = "pink"
    cmaps = {
        "u": shared_cmap,
        "v": shared_cmap,
        "vm": shared_cmap,
    }

    for field, (rel_l2, rel_linf, err) in errs.items():
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
        pcm = ax.pcolormesh(x, y, np.abs(err), cmap=cmaps[field], shading="gouraud")

        skip = 10
        ax.plot(x[::skip, :].T, y[::skip, :].T, "k-", lw=0.4, alpha=0.3)
        ax.plot(x[:, ::skip], y[:, ::skip], "k-", lw=0.4, alpha=0.3)
        ax.plot(x[0, :], y[0, :], "k-", lw=1.2)
        ax.plot(x[-1, :], y[-1, :], "k-", lw=1.2)
        ax.plot(x[:, 0], y[:, 0], "k-", lw=1.2)
        ax.plot(x[:, -1], y[:, -1], "k-", lw=1.2)

        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$", fontsize=11)
        ax.set_ylabel(r"$y$", fontsize=11)
        ax.set_title(
            f"{titles[field]}  (rel $L_2$={rel_l2:.2e}, rel $L_\\infty$={rel_linf:.2e})",
            fontsize=11,
        )

        cbar = fig.colorbar(pcm, ax=ax, pad=0.03)
        cbar.set_label(cbar_labels[field], fontsize=11)

        fig.tight_layout()

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filenames",
        type=str,
        nargs="+",
        help="One or more prediction .npz file paths.",
    )
    parser.add_argument(
        "positional_files",
        nargs="*",
        type=str,
        help="Positional prediction .npz file paths (supports shell globs).",
    )
    parser.add_argument(
        "-r",
        "--ref",
        type=str,
        default="refs/test.npz",
        help="Path to reference solution file (default: refs/test.npz)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Generate error surface plots for each file.",
    )
    args = parser.parse_args()

    all_files = []
    if args.filenames:
        all_files.extend(args.filenames)
    if args.positional_files:
        all_files.extend(args.positional_files)

    if not all_files:
        parser.error(
            "No prediction files provided. Pass via -f file1 file2... or directly as positional arguments."
        )

    for filepath in all_files:
        plot_errs(filepath, args.ref, plot=args.plot)


if __name__ == "__main__":
    main()
