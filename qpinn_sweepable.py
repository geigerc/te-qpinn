"""
And on the fourth day, God said: "Let it converge..."
"""

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import tomllib
from pathlib import Path

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import optax
import pennylane as qml

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


def make_1d_grid(grid_size, ax="x", sampling="uniform", beta=3.0, power=2.5):
    if sampling == "uniform":
        return jnp.linspace(0.0, 1.0, grid_size)

    elif sampling == "chebytto":
        k = jnp.arange(grid_size)
        return 0.5 * (1.0 - jnp.cos(k * jnp.pi / (grid_size - 1)))

    elif sampling == "power":
        return (
            jnp.linspace(0.0, 1.0, grid_size) ** power
            if ax == "x"
            else jnp.linspace(0.0, 1.0, grid_size)
        )

    elif sampling == "asymm_tanh":
        xi = jnp.linspace(0.0, 1.0, grid_size)
        if ax == "x":
            return 1.0 - jnp.tanh(beta * (1.0 - xi)) / jnp.tanh(beta)
        else:
            return 0.5 * (1.0 + jnp.tanh(beta * (2.0 * xi - 1.0)) / jnp.tanh(beta))

    else:
        raise ValueError()


def compute_1d_grid_weights(grid_1d, method="trapezoidal"):
    if method == "trapezoidal":
        dx = jnp.diff(grid_1d)
        w_int = 0.5 * (dx[:-1] + dx[1:])
        w_ends = jnp.array([0.5 * dx[0], 0.5 * dx[-1]])
        return jnp.concat([w_ends[:1], w_int, w_ends[1:]])
    else:
        raise ValueError()


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


def build_qc(cfg, shared=False, unit=None):
    cfg_wires = cfg["qc"]["n_wires"]
    n_wires = 2 * cfg_wires if shared else cfg_wires

    n_layers = cfg["qc"]["n_layers"]
    diff_method = cfg["qc"]["diff_method"]
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

    @qml.qnode(qdev, interface="jax", diff_method=diff_method)
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

        if shared:
            return (
                qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(cfg_wires)])),  # type:ignore
                qml.expval(
                    qml.sum(*[qml.PauliZ(i) for i in range(cfg_wires, n_wires)])  # type:ignore
                ),
            )

        return qml.expval(qml.sum(*[qml.PauliZ(i) for i in range(n_wires)]))  # type: ignore

    return qc


class FNN(eqx.Module):
    layers: tuple

    def __init__(self, cfg, key):
        n_wires = cfg["qc"]["n_wires"]
        depth = cfg["fnn"]["depth"]
        width = cfg["fnn"]["width"]
        activation_fn = cfg["fnn"]["activation"]
        activation = {
            "tanh": jax.nn.tanh,
        }[activation_fn]

        if depth == 0:
            self.layers = (eqx.nn.Linear(2, n_wires, key=key),)
        else:
            keys = jr.split(key, depth + 1)

            layers = [eqx.nn.Linear(2, width, key=keys[0]), activation]
            for k in keys[1:-1]:
                layers += [eqx.nn.Linear(width, width, key=k), activation]
            layers += [eqx.nn.Linear(width, n_wires, key=keys[-1])]

            self.layers = tuple(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class QPINN_unit(eqx.Module):
    theta: jax.Array
    emb: FNN
    qc: callable = eqx.field(static=True)  # type: ignore

    def __init__(self, cfg, key, unit=None):
        keys = jr.split(key, 2)
        n_wires = cfg["qc"]["n_wires"]
        n_layers = cfg["qc"]["n_layers"]

        self.emb = FNN(cfg, keys[0])
        self.theta = jr.uniform(
            keys[1], shape=(n_layers, n_wires, 4), minval=0.0, maxval=2 * jnp.pi
        )
        self.qc = build_qc(cfg, unit=unit)

    def __call__(self, x):
        basis = self.emb(x)
        x_rescaled = 2.0 * x - 1.0
        return self.qc(x_rescaled, self.theta, basis)  # one u component


class QPINN_aggregate(eqx.Module):
    QNN_x: QPINN_unit
    QNN_y: QPINN_unit
    lift_x: jax.Array

    def __init__(self, cfg, key):
        keys = jr.split(key, 2)
        self.QNN_x = QPINN_unit(cfg, keys[0], unit="x")
        self.QNN_y = QPINN_unit(cfg, keys[1], unit="y")
        self.lift_x = jnp.array(0.0)

    def __call__(self, x):
        ux = self.lift_x * x[0] + x[0] * (1.0 - x[0]) * self.QNN_x(x)
        uy = self.QNN_y(x) * x[0]
        return jnp.array([ux, uy])


class QPINN_shared(eqx.Module):
    theta: jax.Array
    emb_x: FNN
    emb_y: FNN
    lift_x: jax.Array
    qc: callable = eqx.field(static=True)  # type:ignore

    def __init__(self, cfg, key):
        keys = jr.split(key, 3)
        n_wires = 2 * cfg["qc"]["n_wires"]
        n_layers = cfg["qc"]["n_layers"]

        self.emb_x = FNN(cfg, keys[0])
        self.emb_y = FNN(cfg, keys[1])
        self.theta = jr.uniform(
            keys[2], shape=(n_layers, n_wires, 4), minval=0.0, maxval=2 * jnp.pi
        )
        self.lift_x = jnp.array(0.0)
        self.qc = build_qc(cfg, shared=True)

    def __call__(self, x):
        basis = jnp.concat([self.emb_x(x), self.emb_y(x)])
        x_rescaled = 2.0 * x - 1.0
        ux_raw, uy_raw = self.qc(x_rescaled, self.theta, basis)

        ux = self.lift_x * x[0] + x[0] * (1.0 - x[0]) * ux_raw
        uy = uy_raw * x[0]
        return jnp.array([ux, uy])


# physics
def pinn_residual(model, x, params):
    del_u = jax.jacobian(model)(x)

    F = jnp.eye(2) + del_u

    C = F.T @ F

    youngs, poisson = params[0], params[1]
    mu = youngs / (2 * (1 + poisson))
    lda = (poisson * youngs) / ((1 + poisson) * (1 - 2 * poisson))

    IC = jnp.trace(C)
    J = jnp.linalg.det(F)

    J_safe = jnp.maximum(J, 1e-4)

    # neo-hookean
    W = lda / 2 * (jnp.log(J_safe)) ** 2 + mu / 2 * (IC - 2 - 2 * jnp.log(J_safe))

    return W, J


def compute_stress(model, x, params):
    del_u = jax.jacobian(model)(x)

    F = jnp.eye(2) + del_u

    youngs, poisson = params[0], params[1]
    mu = youngs / (2 * (1 + poisson))
    lda = (poisson * youngs) / ((1 + poisson) * (1 - 2 * poisson))

    J = jnp.linalg.det(F)

    P = mu * F + (lda * jnp.log(J) - mu) * jnp.linalg.inv(F).T
    sigma = (1.0 / J) * (P @ F.T)

    s11, s22, s12 = sigma[0, 0], sigma[1, 1], sigma[0, 1]
    s33 = (lda * jnp.log(J)) / J

    vm = jnp.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * s12**2
    )

    sigma_3d = jnp.array([[s11, s12, 0.0], [s12, s22, 0.0], [0.0, 0.0, s33]])

    return vm, sigma_3d


# losses
def make_loss_fn(params, cfg):
    lda_penalty_j = cfg["loss"]["lda_penalty_j"]
    lda_neumann = cfg["loss"]["lda_neumann"]
    lda_physics = cfg["loss"]["lda_physics"]

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(model, x_boundary, x_physics, w_physics, traction):

        u_b_pred = jax.vmap(model)(x_boundary)
        ux_b_pred = u_b_pred[:, 0]

        energy_potentials, Js = jax.vmap(lambda x: pinn_residual(model, x, params))(
            x_physics
        )

        loss_neumann = lda_neumann * -jnp.trapezoid(
            traction * ux_b_pred, x_boundary[:, 1]
        )
        # penalty for negative Js
        loss_barrier = lda_penalty_j * jnp.sum(jax.nn.relu(0.05 - Js) ** 2 * w_physics)

        loss_physics = lda_physics * jnp.sum(energy_potentials * w_physics)

        total_loss = loss_neumann + loss_physics + loss_barrier

        return total_loss, (loss_neumann, loss_physics, loss_barrier, Js)

    return loss_fn


def main(cfg, output_path, cfg_path):

    seed = cfg["meta"]["seed"]
    run_name = cfg["meta"]["run_name"]

    youngs = cfg["material"]["youngs"]
    poisson = cfg["material"]["poisson"]
    params = [youngs, poisson]

    traction_x = cfg["loading"]["traction_x"]
    n_stages = cfg["loading"]["n_stages"]
    step_epochs = cfg["loading"]["step_epochs"]

    grid_size = cfg["domain"]["grid_size"]
    sampling = cfg["domain"]["sampling"]

    init_lr = cfg["optimizer"]["init_lr"]
    transition_steps = cfg["optimizer"]["transition_steps"]
    decay_rate = cfg["optimizer"]["decay_rate"]

    model_key = jr.PRNGKey(seed)

    mode = cfg["architecture"]["mode"]
    model = {"separate": QPINN_aggregate, "shared": QPINN_shared}[mode](cfg, model_key)

    _x = make_1d_grid(grid_size, ax="x", sampling=sampling)
    _y = make_1d_grid(grid_size, ax="y", sampling=sampling)

    wx = compute_1d_grid_weights(_x)
    wy = compute_1d_grid_weights(_y)

    X, Y = jnp.meshgrid(_x, _y)
    WX, WY = jnp.meshgrid(wx, wy)

    x_physics = jnp.column_stack([X.ravel(), Y.ravel()])
    w_physics = (WX * WY).ravel()

    x_boundary_right = jnp.column_stack([jnp.ones(grid_size), _y])

    # backend check
    print("active backend:", jax.default_backend())
    print("detected devices:", jax.devices())

    lr_schedule = optax.exponential_decay(
        init_value=init_lr, transition_steps=transition_steps, decay_rate=decay_rate
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_fn = make_loss_fn(params, cfg)

    # jax.lax.scan for XLA-d for loops
    @eqx.filter_jit
    def train(
        model,
        opt_state,
        x_boundary_right,
        x_physics,
        w_physics,
        traction,
        n_epochs,
    ):
        params, static = eqx.partition(model, eqx.is_array)

        def body(carry, _):

            params, opt_state = carry
            model = eqx.combine(params, static)
            (total_loss, (loss_neumann, loss_physics, loss_barrier, Js)), grads = (
                loss_fn(model, x_boundary_right, x_physics, w_physics, traction)
            )
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            params, _ = eqx.partition(model, eqx.is_array)
            return (params, opt_state), (
                total_loss,
                loss_neumann,
                loss_physics,
                loss_barrier,
                Js,
            )

        (
            (params, opt_state),
            (total_loss, loss_neumann, loss_physics, loss_barrier, Js),
        ) = jax.lax.scan(body, (params, opt_state), None, length=n_epochs)
        model = eqx.combine(params, static)
        return (
            model,
            opt_state,
            total_loss,
            loss_neumann,
            loss_physics,
            loss_barrier,
            Js,
        )

    # training loop
    loss_history = []

    for stage in range(n_stages):
        scaling_factor = (stage + 1) / n_stages

        t_incr = jnp.asarray(traction_x * scaling_factor)

        model, opt_state, total_loss, loss_neumann, loss_physics, loss_barrier, Js = (
            train(
                model,
                opt_state,
                x_boundary_right,
                x_physics,
                w_physics,
                t_incr,
                step_epochs,
            )
        )
        loss_history.append((total_loss, loss_neumann, loss_physics, loss_barrier))

        print(
            f"stage={stage + 1:02d} | scale={scaling_factor:.2f} | loss: {total_loss[-1]:.6f} "
            f"| neumann={loss_neumann[-1]:.4f} phys={loss_physics[-1]:.4f} barrier={loss_barrier[-1]:.4f} "
            f"| min J {Js.min():.4f} | max J {Js.max():.4f}"
        )

    # saving
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    eqx.tree_serialise_leaves(str(output / f"{run_name}.eqx"), model)

    import shutil

    shutil.copy(cfg_path, output / f"{run_name}_config.toml")

    total_h, neumann_h, physics_h, barrier_h = zip(*loss_history)
    jnp.savez(
        output / f"{run_name}_loss_history.npz",
        total=jnp.array(total_h),
        neumann=jnp.array(neumann_h),
        physics=jnp.array(physics_h),
        barrier=jnp.array(barrier_h),
    )

    # uniform grid for fair comparison against ref
    _x_e = make_1d_grid(grid_size, ax="x", sampling="uniform")
    _y_e = make_1d_grid(grid_size, ax="y", sampling="uniform")
    X_eval, Y_eval = jnp.meshgrid(_x_e, _y_e)
    x_eval = jnp.column_stack([X_eval.ravel(), Y_eval.ravel()])

    u_pred = jax.vmap(model)(x_eval)
    vm_pred, sigma_3d_pred = jax.vmap(lambda x: compute_stress(model, x, params))(
        x_eval
    )

    u_grid = u_pred[:, 0].reshape(grid_size, grid_size)
    v_grid = u_pred[:, 1].reshape(grid_size, grid_size)
    vm_grid = vm_pred.reshape(grid_size, grid_size)

    u_grid = u_grid.T
    v_grid = v_grid.T
    vm_grid = vm_grid.T

    jnp.savez(
        output / f"{run_name}_prediction.npz",
        # do not confuse downstream
        x_1d=_x_e,
        y_1d=_y_e,
        x_train_1d=_x,
        y_train_1d=_y,
        u_grid=u_grid,
        v_grid=v_grid,
        von_mises_grid=vm_grid,
        sigma_3d_pred=sigma_3d_pred,
        x_physics=jnp.array(x_physics),
        x_eval=jnp.array(x_eval),
        u_pred=jnp.array(u_pred),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)

    main(cfg, args.output, args.config)
