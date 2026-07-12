# pennylane, jax
# diffrax, optax, equinox

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

POISSON = 0.3
YOUNGS = 1
BC_LOSS_WEIGHT = 50
EPOCHS = 20000


class PINN(eqx.Module):
    layers: tuple

    def __init__(self, key):
        keys = jr.split(key, 5)
        self.layers = (
            eqx.nn.Linear(2, 32, key=keys[0]),
            jax.nn.tanh,
            eqx.nn.Linear(32, 32, key=keys[1]),
            jax.nn.tanh,
            eqx.nn.Linear(32, 32, key=keys[2]),
            jax.nn.tanh,
            eqx.nn.Linear(32, 32, key=keys[3]),
            jax.nn.tanh,
            eqx.nn.Linear(32, 2, key=keys[4]),
        )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# solving for predicted displacement vector u,
# given input vectors x, to get derivatives
# and buid deformation vector F = I + del u ;
# then C;
# then compute energy potential term which
# serves as loss(minus the dirichlet term)
# and then backprop


# physics derivatives
def pinn_residual(model, x):
    del_u = jax.jacobian(model)(x)

    F = jnp.eye(2) + del_u

    C = jnp.dot(F.T, F)

    mu = YOUNGS / (2 * (1 + POISSON))
    lda = (POISSON * YOUNGS) / ((1 + POISSON) * (1 - 2 * POISSON))
    IC = jnp.trace(C)
    J = jnp.linalg.det(F)

    # softplus = lambda x: jnp.logaddexp(0, x)
    # def safe_J(J, eps=1e-2):
    #     return jnp.where(J > eps, J, eps * jnp.exp((J - eps) / eps))

    # J_safe = safe_J(J)
    # J_safe = jnp.maximum(J, 1e-4)

    # def torchlikeSoftplus(x, beta=1.0, threshold=20.0):
    #     bx = beta * x
    #     return jnp.where(bx > threshold, x, 1 / beta * jnp.logaddexp(0, beta * x))

    def torchlike_softplus(x, beta=1.0):
        return jnp.logaddexp(0, beta * x) / beta

    J_safe = torchlike_softplus(J, beta=10)

    # TODO: fix this. gradients die with many epochs

    W = lda / 2 * (jnp.log(J_safe)) ** 2 + mu / 2 * (IC - 2 - 2 * jnp.log(J_safe))

    return W


# losses
@eqx.filter_value_and_grad
def loss_fn(model, x_boundary, u_boundary, x_physics):
    u_b_pred = jax.vmap(model)(x_boundary)
    loss_data = jnp.mean(jnp.sum((u_b_pred - u_boundary) ** 2, axis=-1))

    energy_potentials = jax.vmap(lambda x: pinn_residual(model, x))(x_physics)
    loss_physics = jnp.mean(energy_potentials)

    return (BC_LOSS_WEIGHT * loss_data) + loss_physics


def main():

    # init
    key = jr.PRNGKey(42)
    model_key, training_key = jr.split(key)

    model = PINN(model_key)

    _x = jnp.linspace(0, 1, 10)
    _y = jnp.linspace(0, 1, 10)
    X, Y = jnp.meshgrid(_x, _y)
    x_physics = jnp.column_stack([X.ravel(), Y.ravel()])

    # left boundary clamped
    x_boundary_left = jnp.column_stack([jnp.zeros(10), jnp.linspace(0, 1, 10)])
    u_boundary_left = jnp.zeros_like(x_boundary_left)

    # right boundary
    x_boundary_right = jnp.column_stack([jnp.ones(10), jnp.linspace(0, 1, 10)])
    u_boundary_right = jnp.column_stack([jnp.ones(10) * 0.5, jnp.zeros(10)])

    x_boundary = jnp.vstack([x_boundary_left, x_boundary_right])
    u_boundary = jnp.vstack([u_boundary_left, u_boundary_right])

    initial_lr = 0.002
    lr_schedule = optax.exponential_decay(
        init_value=initial_lr, transition_steps=2000, decay_rate=0.5
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # optimization step
    @eqx.filter_jit
    def step(model, opt_state, x_b, u_b, x_p):
        loss, grads = loss_fn(model, x_b, u_b, x_p)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # training loop
    loss_history = []
    for epoch in range(EPOCHS + 1):
        model, opt_state, loss = step(
            model, opt_state, x_boundary, u_boundary, x_physics
        )
        loss_history.append(loss)
        if epoch % 200 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss:.6f}")

    eqx.tree_serialise_leaves("pinn.eqx", model)

    u_pred = jax.vmap(model)(x_physics)

    x_deformed = x_physics + u_pred

    X_init = x_physics[:, 0].reshape(10, 10)
    Y_init = x_physics[:, 1].reshape(10, 10)

    X_def = x_deformed[:, 0].reshape(10, 10)
    Y_def = x_deformed[:, 1].reshape(10, 10)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(10):
        ax[0].plot(X_init[i, :], Y_init[i, :], "k:", alpha=0.4, zorder=1)
        ax[0].plot(X_init[:, i], Y_init[:, i], "k:", alpha=0.4, zorder=1)

    ax[0].scatter(
        x_physics[:, 0],
        x_physics[:, 1],
        color="grey",
        s=10,
        alpha=0.3,
        label="Reference Nodes",
        zorder=1,
    )

    for i in range(10):
        ax[0].plot(X_def[i, :], Y_def[i, :], "r-", alpha=0.7, zorder=2)
        ax[0].plot(X_def[:, i], Y_def[:, i], "r-", alpha=0.7, zorder=2)

    ax[0].scatter(
        x_deformed[:, 0],
        x_deformed[:, 1],
        color="red",
        s=18,
        label="Deformed Nodes",
        zorder=2,
    )

    ax[0].set_title("Deformation")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_xlim(-0.1, 2)
    ax[0].set_ylim(-0.1, 1.5)
    ax[0].grid(True, linestyle=":", alpha=0.5)
    ax[0].legend(loc="upper left")

    ax[1].plot(loss_history, color="darkblue", linewidth=1.8, label="Total Loss")
    ax[1].set_yscale("log")
    ax[1].set_title("Loss History")
    ax[1].set_xlabel("Training Epochs")
    ax[1].set_ylabel("Loss Metric (Log Scale)")
    ax[1].grid(True, which="both", linestyle=":", alpha=0.5)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


# TODO: split into modeldefinition py, training py, query py or just repl shell

if __name__ == "__main__":
    main()
