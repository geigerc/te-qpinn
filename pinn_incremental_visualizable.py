"""
Converge...is our goal...
"""

# pennylane, jax
# diffrax, optax, equinox
import os

# lets jax allocate memory dynamically thru cudaMalloc
# instead of grabbing 90% OF MY VRAM
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# alternatively
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

POISSON = 0.3
YOUNGS = 1
BC_LOSS_WEIGHT = 10000
STEP_EPOCHS = 500


class PINN(eqx.Module):
    layers: tuple

    # signature: (x, u_bc_vector) -> u
    def __init__(self, key):
        keys = jr.split(key, 5)
        self.layers = (
            eqx.nn.Linear(4, 32, key=keys[0]),
            jax.nn.tanh,
            eqx.nn.Linear(32, 32, key=keys[1]),
            jax.nn.tanh,
            eqx.nn.Linear(32, 32, key=keys[2]),
            jax.nn.tanh,
            eqx.nn.Linear(32, 32, key=keys[3]),
            jax.nn.tanh,
            eqx.nn.Linear(32, 2, key=keys[4]),
        )

    def __call__(self, x, u_bc_locator):
        input = jnp.concatenate([x, u_bc_locator])
        for layer in self.layers:
            input = layer(input)
        return input


# solving for predicted displacement vector u,
# given input vectors x, to get derivatives
# and buid deformation vector F = I + del u ;
# then C;
# then compute energy potential term which
# serves as loss(minus the dirichlet term)
# and then backprop


# physics derivatives
def pinn_residual(model, x, u_bc_locator):
    del_u = jax.jacobian(lambda xx: model(xx, u_bc_locator))(x)

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
    J_safe = jnp.maximum(J, 1e-4)

    # def torchlikeSoftplus(x, beta=1.0, threshold=20.0):
    #     bx = beta * x
    #     return jnp.where(bx > threshold, x, 1 / beta * jnp.logaddexp(0, beta * x))

    # def torchlike_softplus(x, beta=1.0):
    #     return jnp.logaddexp(0, beta * x) / beta

    # J_safe = torchlike_softplus(J, beta=10)

    W = lda / 2 * (jnp.log(J_safe)) ** 2 + mu / 2 * (IC - 2 - 2 * jnp.log(J_safe))

    return W, J


# losses
@eqx.filter_value_and_grad(has_aux=True)
def loss_fn(
    model, x_boundary_bot, u_boundary_bot, x_boundary_top, x_physics, u_bc_batch
):

    def single_state_loss(u_bc_locator):
        u_boundary_top = jnp.broadcast_to(u_bc_locator, x_boundary_top.shape)
        x_boundary = jnp.vstack([x_boundary_bot, x_boundary_top])
        u_boundary = jnp.vstack([u_boundary_bot, u_boundary_top])

        u_b_pred = jax.vmap(lambda x: model(x, u_bc_locator))(x_boundary)
        loss_data = jnp.mean(jnp.sum((u_b_pred - u_boundary) ** 2, axis=-1))

        energy_potentials, Js = jax.vmap(
            lambda x: pinn_residual(model, x, u_bc_locator)
        )(x_physics)

        # penalty for negative Js
        penalty_weight_J = 10000
        loss_barrier = jnp.mean(jax.nn.relu(0.05 - Js) ** 2) * penalty_weight_J

        loss_physics = jnp.mean(energy_potentials)

        return (BC_LOSS_WEIGHT * loss_data) + loss_physics + loss_barrier, Js

    losses, Js_batch = jax.vmap(single_state_loss)(u_bc_batch)

    return jnp.mean(losses), Js_batch


def main():

    # init
    key = jr.PRNGKey(888)  # prev 42
    model_key, training_key = jr.split(key)

    model = PINN(model_key)

    _x = jnp.linspace(0, 1, 50)
    _y = jnp.linspace(0, 1, 50)
    X, Y = jnp.meshgrid(_x, _y)
    x_physics = jnp.column_stack([X.ravel(), Y.ravel()])

    # bottom boundary clamped
    x_boundary_bot = jnp.column_stack([jnp.linspace(0, 1, 50), jnp.zeros(50)])
    u_boundary_bot = jnp.zeros_like(x_boundary_bot)
    x_boundary_top = jnp.column_stack([jnp.linspace(0, 1, 50), jnp.ones(50)])

    # sampling box
    u_bc_box_bot = jnp.column_stack([jnp.linspace(0, 1, 50), jnp.ones(50) * 0.8])
    u_bc_box_top = jnp.column_stack([jnp.linspace(0, 1, 50), jnp.ones(50) * 1.6])
    u_bc_box_left = jnp.column_stack(
        [jnp.ones(50) * (-0.5), jnp.linspace(0.5, 1.6, 50)]
    )
    u_bc_box_right = jnp.column_stack(
        [jnp.ones(50) * (1.5), jnp.linspace(0.5, 1.6, 50)]
    )

    u_bc_box = jnp.vstack([u_bc_box_bot, u_bc_box_top, u_bc_box_left, u_bc_box_right])

    # backend check
    print("active backend:", jax.default_backend())
    print("detected devices:", jax.devices())

    initial_lr = 0.002
    lr_schedule = optax.exponential_decay(
        init_value=initial_lr, transition_steps=300000, decay_rate=0.5
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # jax.lax.scan implementation of python for loop,
    # to bypass python involvement and run everything
    # directly on gpu
    @eqx.filter_jit
    def train(
        model,
        opt_state,
        x_boundary_bot,
        u_boundary_bot,
        x_boundary_top,
        x_physics,
        u_bc_batch,
        n_epochs,
    ):
        params, static = eqx.partition(model, eqx.is_array)

        def body(carry, _):

            params, opt_state = carry
            model = eqx.combine(params, static)
            (losses, Js), grads = loss_fn(
                model,
                x_boundary_bot,
                u_boundary_bot,
                x_boundary_top,
                x_physics,
                u_bc_batch,
            )
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            params, _ = eqx.partition(model, eqx.is_array)
            return (params, opt_state), (losses, Js)

        (params, opt_state), (losses, Js) = jax.lax.scan(
            body, (params, opt_state), None, length=n_epochs
        )
        model = eqx.combine(params, static)

        return model, opt_state, losses, Js

    # note: jax.lax.scan falttens any pytree given,
    # and tanh is considered a leaf, but is a function
    # not an array, so first the model must be split
    # into parameters and static parts, otherwise jax
    # returns type error because it cant work on fns

    # training loop

    anchor = x_physics[2475]  # [0.5102, 1.0]
    N_STAGES = 60
    BATCH_SIZE = 32

    key = training_key

    loss_history = []
    # cumulative_epoch = 0

    for stage in range(N_STAGES):
        key, subkey = jr.split(key)
        u_bc_sample = u_bc_box[
            jr.choice(subkey, u_bc_box.shape[0], shape=(BATCH_SIZE,))
        ]
        scaling_factor = (stage + 1) / N_STAGES

        u_bc_batch = (u_bc_sample - anchor) * scaling_factor

        model, opt_state, loss, Js = train(
            model,
            opt_state,
            x_boundary_bot,
            u_boundary_bot,
            x_boundary_top,
            x_physics,
            u_bc_batch,
            STEP_EPOCHS,
        )

        loss_history.append(loss)

        print(
            f"stage={stage + 1:02d} | scale={scaling_factor:.2f} | loss: {loss[-1]:.6f} | min J {Js.min():.4f} | max J {Js.max():.4f}"
        )

    # saving
    eqx.tree_serialise_leaves("pinn_incremental.eqx", model)
    jnp.save("loss_history.npy", jnp.array(loss_history))

    eval_u = u_bc_box_top[-1] - anchor
    u_pred = jax.vmap(lambda x: model(x, eval_u))(x_physics)

    from plotterer import plot_results

    loss_history = jnp.concatenate(loss_history)
    plot_results(x_physics, u_pred, grid_dim=50, loss_history=loss_history)


# TODO: split into modeldefinition py, training py, query py or just repl shell

if __name__ == "__main__":
    main()
