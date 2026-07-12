import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from pinn_incremental import PINN


def plot_results(x_physics, u_pred, grid_dim, loss_history=None):
    x_deformed = x_physics + u_pred

    X_init = x_physics[:, 0].reshape(grid_dim, grid_dim)
    Y_init = x_physics[:, 1].reshape(grid_dim, grid_dim)

    X_def = x_deformed[:, 0].reshape(grid_dim, grid_dim)
    Y_def = x_deformed[:, 1].reshape(grid_dim, grid_dim)

    if loss_history is not None:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(7, 6))
        ax1 = None

    for i in range(grid_dim):
        ax0.plot(X_init[i, :], Y_init[i, :], "k:", alpha=0.4, zorder=1)
        ax0.plot(X_init[:, i], Y_init[:, i], "k:", alpha=0.4, zorder=1)

    ax0.scatter(
        x_physics[:, 0],
        x_physics[:, 1],
        color="grey",
        s=10,
        alpha=0.3,
        label="Reference Nodes",
        zorder=1,
    )

    for i in range(grid_dim):
        ax0.plot(X_def[i, :], Y_def[i, :], "r-", alpha=0.7, zorder=2)
        ax0.plot(X_def[:, i], Y_def[:, i], "r-", alpha=0.7, zorder=2)

    ax0.scatter(
        x_deformed[:, 0],
        x_deformed[:, 1],
        color="red",
        s=18,
        label="Deformed Nodes",
        zorder=2,
    )

    ax0.set_title("Deformation")
    ax0.set_xlabel("X")
    ax0.set_ylabel("Y")
    ax0.set_xlim(-0.1, 2)
    ax0.set_ylim(-0.1, 2)
    ax0.grid(True, linestyle=":", alpha=0.5)
    ax0.legend(loc="upper left")

    if ax1 is not None:
        ax1.plot(loss_history, color="darkblue", linewidth=1.8, label="Total Loss")
        ax1.set_yscale("log")
        ax1.set_title("Loss History")
        ax1.set_xlabel("Training Epochs")
        ax1.set_ylabel(r"$\mathcal{L}_\psi + L_{2,\text{D}}$ (Log Scale)")
        ax1.grid(True, which="both", linestyle=":", alpha=0.5)
        ax1.legend()

    plt.tight_layout()
    plt.show()


def main():

    skeleton = PINN(jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves("pinn_incremental.eqx", skeleton)

    _x = jnp.linspace(0, 1, 10)
    _y = jnp.linspace(0, 1, 10)
    X, Y = jnp.meshgrid(_x, _y)
    x_physics = jnp.column_stack([X.ravel(), Y.ravel()])

    u_pred = jax.vmap(model)(x_physics)

    loss_history = jnp.load("loss_history.npy")

    plot_results(x_physics, u_pred, grid_dim=10, loss_history=loss_history)


if __name__ == "__main__":
    main()
