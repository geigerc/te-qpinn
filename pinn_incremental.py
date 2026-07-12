# pennylane, jax
# diffrax, optax, equinox
import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

POISSON = 0.3
YOUNGS = 1
BC_LOSS_WEIGHT = 1000
STEP_EPOCHS = 500


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

    W = lda / 2 * (jnp.log(J_safe)) ** 2 + mu / 2 * (IC - 2 - 2 * jnp.log(J_safe))

    return W, J


# losses
@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(model, x_boundary, u_boundary, x_physics):
    u_b_pred = jax.vmap(model)(x_boundary)
    loss_data = jnp.mean(jnp.sum((u_b_pred - u_boundary) ** 2, axis=-1))

    energy_potentials, Js = jax.vmap(lambda x: pinn_residual(model, x))(x_physics)
    loss_physics = jnp.mean(energy_potentials)

    return (BC_LOSS_WEIGHT * loss_data) + loss_physics, Js


def main():

    # init
    key = jr.PRNGKey(888)  # prev 42
    model_key, training_key = jr.split(key)

    model = PINN(model_key)

    _x = jnp.linspace(0, 1, 10)
    _y = jnp.linspace(0, 1, 10)
    X, Y = jnp.meshgrid(_x, _y)
    x_physics = jnp.column_stack([X.ravel(), Y.ravel()])

    # bottom boundary clamped
    x_boundary_bot = jnp.column_stack([jnp.linspace(0, 1, 10), jnp.zeros(10)])
    u_boundary_bot = jnp.zeros_like(x_boundary_bot)

    x_boundary_top = jnp.column_stack([jnp.linspace(0, 1, 10), jnp.ones(10)])
    x_boundary = jnp.vstack([x_boundary_bot, x_boundary_top])

    # left boundary clamped
    # x_boundary_left = jnp.column_stack([jnp.zeros(10), jnp.linspace(0, 1, 10)])
    # u_boundary_left = jnp.zeros_like(x_boundary_left)

    # # define incremental u_boundary later
    # x_boundary_right = jnp.column_stack([jnp.ones(10), jnp.linspace(0, 1, 10)])
    # x_boundary = jnp.vstack([x_boundary_left, x_boundary_right])

    # backend check
    print("active backend:", jax.default_backend())
    print("detected devices:", jax.devices())

    initial_lr = 0.002
    lr_schedule = optax.exponential_decay(
        init_value=initial_lr, transition_steps=2000, decay_rate=0.5
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # optimization step
    @eqx.filter_jit
    def step(model, opt_state, x_b, u_b, x_p):
        (loss, Js), grads = loss_fn(model, x_b, u_b, x_p)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, Js

    # training loop

    angle = jnp.deg2rad(30)  # for shear

    loss_history = []
    cumulative_epoch = 0
    for step_idx, incr_disp in enumerate(jnp.linspace(0.05, 1, 60)):
        # top bot config
        u_boundary_top = jnp.column_stack(
            [
                jnp.ones(10) * incr_disp * jnp.cos(angle),
                jnp.ones(10) * incr_disp * jnp.sin(angle),
            ]
        )
        u_boundary = jnp.vstack([u_boundary_bot, u_boundary_top])

        # left right config
        # u_boundary_right = jnp.column_stack([jnp.ones(10) * incr_disp, jnp.zeros(10)])
        # u_boundary = jnp.vstack([u_boundary_left, u_boundary_right])

        for epoch in range(STEP_EPOCHS + 1):
            model, opt_state, loss, Js = step(
                model, opt_state, x_boundary, u_boundary, x_physics
            )
            loss_history.append(loss)

            if cumulative_epoch % 200 == 0:
                print(
                    f"disp={incr_disp:.2f} | step={step_idx + 1:02d}| epoch {cumulative_epoch:05d} | loss: {loss:.6f} | min J {Js.min():.4f} | max J {Js.max():.4f}"
                )

            cumulative_epoch += 1

    # saving
    # eqx.tree_serialise_leaves("pinn_incremental.eqx", model)
    # jnp.save("loss_history.npy", jnp.array(loss_history))

    u_pred = jax.vmap(model)(x_physics)

    from plotterer import plot_results

    plot_results(x_physics, u_pred, loss_history=loss_history)


# TODO: split into modeldefinition py, training py, query py or just repl shell

if __name__ == "__main__":
    main()
