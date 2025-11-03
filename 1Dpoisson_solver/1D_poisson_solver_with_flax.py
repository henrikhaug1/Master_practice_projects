import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax
from flax import linen as nn
import matplotlib.pyplot as plt

# Problem: -u''(x) = f(x) on [0,1], u(0)=u(1)=0
key = jax.random.PRNGKey(0)
pi = jnp.pi


def f(x):
    return (pi**2) * jnp.sin(pi * x)


class NN(nn.Module):
    widths: tuple

    @nn.compact
    def __call__(self, x):
        z = x
        for w in self.widths:
            z = nn.tanh(nn.Dense(w)(z))
        return nn.Dense(1)(z)


model = NN(widths=(32, 32))
params = model.init(key, jnp.ones((1, 1)))["params"]


def n_apply(params, x):
    x2d = x[:, None]
    return model.apply({"params": params}, x2d)[:, 0]


def u_hat(params, x):
    return x * (1.0 - x) * n_apply(params, x)


def _u_scalar(params, t):
    return u_hat(params, jnp.array([t]))[0]


def u_x(params, x):
    return jax.jacfwd(lambda t: _u_scalar(params, t))(x)


def u_xx(params, x):
    return jax.jacfwd(jax.jacrev(lambda t: _u_scalar(params, t)))(x)


def u_xx_vec(params, x_vec):
    return jax.vmap(lambda t: u_xx(params, t))(x_vec)


def residual(params, x):
    return -u_xx_vec(params, x) - f(x)


def loss_fn(params, x):
    r = residual(params, x)
    return jnp.mean(r**2)


opt = optax.adam(1e-3)
opt_state = opt.init(params)


@jax.jit
def train_step(params, opt_state, x):
    loss, grads = jax.value_and_grad(loss_fn)(params, x)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


N = 256
key, kx = jax.random.split(key)
x_train = jax.random.uniform(kx, (N,), minval=0.0, maxval=1.0)

# Train
loss_hist = []
for it in range(4000):
    params, opt_state, L = train_step(params, opt_state, x_train)
    loss_hist.append(float(L))
    if (it + 1) % 500 == 0:
        print(f"iter {it + 1:4d}  loss={L:.3e}")

x_eval = jnp.linspace(0, 1, 200)
u_pred = u_hat(params, x_eval)
u_true = jnp.sin(pi * x_eval)

rel_L2 = float(jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true))
print("Relative L2 error:", f"{rel_L2:.3e}")

# Plot
plt.figure()
plt.plot(x_eval, u_pred, label="PINN")
plt.plot(x_eval, u_true, "--", label="True")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Solution")
plt.legend()
plt.grid(True)

uxx = u_xx_vec(params, x_eval)
res = -uxx - f(x_eval)
plt.figure()
plt.plot(x_eval, res)
plt.xlabel("x")
plt.ylabel(r"$-u''_\theta - f$")
plt.title("PDE residual")
plt.grid(True)

plt.figure()
plt.plot(x_eval, jnp.abs(u_pred - u_true))
plt.xlabel("x")
plt.ylabel("|error|")
plt.title("Absolute error")
plt.grid(True)

plt.figure()
plt.semilogy(loss_hist)
plt.xlabel("iteration")
plt.ylabel("physics loss")
plt.title("Training loss")
plt.grid(True)
plt.show()
