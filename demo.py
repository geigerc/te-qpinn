import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from pinn_incremental_visualizable import PINN


def load_model(trained_path, key=888):
    key = jr.PRNGKey(key)
    model_key, _ = jr.split(key)
    skeleton = PINN(model_key)
    model = eqx.tree_deserialise_leaves(trained_path, skeleton)

    return model


def main():
    model = load_model("pinn_incremental.eqx")

    # grid
    _x = jnp.linspace(0, 1, 50)
    _y = jnp.linspace(0, 1, 50)
    X, Y = jnp.meshgrid(_x, _y)
    x_physics = jnp.column_stack([X.ravel(), Y.ravel()])

    anchor = x_physics[2475]

    @jax.jit
    def predict(u_bc_locator):
        u_pred = jax.vmap(lambda x: model(x, u_bc_locator))(x_physics)
        return x_physics + u_pred

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-0.5, 3.0)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.scatter(x_physics[:, 0], x_physics[:, 1], c="lightgray", s=5, alpha=0.5)

    active_mesh = ax.scatter(x_physics[:, 0], x_physics[:, 1], c="red", s=5)

    (handle,) = ax.plot(
        [anchor[0]], [anchor[1]], "bo", markersize=12, picker=10, label="DRAG ME"
    )
    ax.legend(loc="upper left")

    drag_state = {"is_dragging": False}

    def on_press(event):
        if event.inaxes != ax:
            return
        contains, _ = handle.contains(event)
        if contains:
            drag_state["is_dragging"] = True

    def on_release(event):
        drag_state["is_dragging"] = False

    def on_motion(event):
        if not drag_state["is_dragging"] or event.inaxes != ax:
            return

        new_x, new_y = event.xdata, event.ydata
        handle.set_data([new_x], [new_y])

        u_bc_locator = jnp.array([new_x, new_y]) - anchor

        new_coords = predict(u_bc_locator)

        active_mesh.set_offsets(np.array(new_coords))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    predict(jnp.array([0.0, 0.0]))

    plt.show()


if __name__ == "__main__":
    main()
