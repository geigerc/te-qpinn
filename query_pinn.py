import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from pinn import PINN, pinn_residual

skeleton = PINN(jr.PRNGKey(0))
model = eqx.tree_deserialise_leaves("pinn_incremental.eqx", skeleton)

# F-I~0 check
print(pinn_residual(model, jnp.array([0.0, 0.5])))

# min J check
xs = jnp.linspace(0, 1, 50)
ys = jnp.linspace(0, 1, 50)
X, Y = jnp.meshgrid(xs, ys)
grid = jnp.column_stack([X.ravel(), Y.ravel()])


def get_J(model, x):
    del_u = jax.jacobian(model)(x)
    F = jnp.eye(2) + del_u
    return jnp.linalg.det(F)


Js = jax.vmap(lambda x: get_J(model, x))(grid)
print("min J:", Js.min(), "max J:", Js.max())
# min J: 1.1134516 max J: 1.7107866

# bc check
print(model(jnp.array([0.0, 0.5])))
print(model(jnp.array([1.0, 0.5])))
# [ 0.00675855 -0.00014951]
# [ 0.493733  -0.0006365]


# symmetry check
print(model(jnp.array([0.5, 0.2])))
print(model(jnp.array([0.5, 0.8])))
# [0.2630419  0.04937714]
# [ 0.26457083 -0.04501186]

# free surface check
for y in [0.0, 0.02, 0.98, 1.0]:
    print(y, get_J(model, jnp.array([0.5, y])))
# 0.0 1.2695941
# 0.02 1.2732241
# 0.98 1.199431
# 1.0 1.1935692


# incremental model (10 increments):
# ❯ uv run .\query_pinn.py
# 0.05043572
# min J: 1.2181444 max J: 1.3478833
# [ 0.0069047  -0.00033619]
# [ 4.9480838e-01 -8.4359199e-06]
# [0.25247484 0.05096544]
# [ 0.25319928 -0.05434912]
# 0.0 1.3049861
# 0.02 1.3069732
# 0.98 1.300891
# 1.0 1.2971685

# incremental model (20 steps | 1000 epochs/step )
# ❯ uv run .\query_pinn.py
# 0.049357384
# min J: 1.2210686 max J: 1.3468422
# [ 0.00701591 -0.00022467]
# [4.9476793e-01 3.8621947e-05]
# [0.25205728 0.05151749]
# [ 0.25263172 -0.05421935]
# 0.0 1.3022934
# 0.02 1.3045652
# 0.98 1.3033776
# 1.0 1.3001808

# incremental model (40 steps | 500 epochs/step)
# ❯ uv run .\query_pinn.py
# 0.048729718
# min J: 1.226183 max J: 1.3504453
# [ 0.0069102  -0.00019185]
# [4.9473152e-01 2.5434420e-05]
# [0.25256893 0.051548  ]
# [ 0.25273475 -0.05259805]
# 0.0 1.3085147
# 0.02 1.3106054
# 0.98 1.311706
# 1.0 1.3089252
