import jax, jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

key = jax.random.PRNGKey(0)
pi = jnp.pi
f = lambda x: (pi**2) * jnp.sin(pi * x)


def init_FFNN(key, layers):
    params = []
    keys = jax.random.split(key, len(layers) - 1)
    for k, (m, n) in zip(
        keys, zip(layers[:-1], layers[1:])
    ):  # will match 1 key with each pair like (key1, (layer1,layer2), (key2, (layer3, layer4))) etc...
        w = jax.random.normal(k, (m, n)) / jnp.sqrt(m)
        b = jnp.zeros((n,))
        params.append((w, b))
    return params


def FFNN_forward(params, x):
    z = x
    for w, b in params[:-1]:  # for all hidden layers
        z = jnp.tanh(z @ w + b)  # put through activation function
    wL, bL = params[-1]  # output
    return z @ wL + bL  # no activation for output


def u_hat(params, x):
    n = FFNN_forward(params, x[:, None])[:, 0]
    return x * (1.0 - x) * n  # This enforces dirichlet boundary conditions u(0)=u(1)=0


def u_x(params, x):
    return jax.grad(lambda t: u_hat(params, jnp.array([t]))[0])(x)


def u_xx(params, x):
    return jax.grad(lambda t: u_x(params, t))(x)


v_u_xx = jax.vmap(
    lambda t, p: u_xx(p, t), in_axes=(0, None)
)  # vectorize derivation with vmap --> makes a vectorized input


def residual(params, x):
    res = v_u_xx(x, params) - f(x)
    return res


def loss_func(params, x):
    res = residual(params, x)
    return jnp.mean(res**2)


# ---------- TRAINING ----------
layers = [1, 32, 32, 1]
params = init_FFNN(key, layers)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)


@jax.jit
def train(params, opt_state, x):
    loss, grads = jax.value_and_grad(loss_func)(params, x)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


N = 256
key, kx = jax.random.split(key)  # new RNG key
x_train = jax.random.uniform(
    kx, (N,), minval=0.0, maxval=1.0
)  # random sample points in interval
loss_hist = []
# ---- Optimization loop ----
for it in range(4000):
    params, opt_state, L = train(params, opt_state, x_train)
    loss_hist.append(float(L))
    if (it + 1) % 500 == 0:  # log every 500 iters
        print(f"iter {it + 1:4d}  loss={L:.3e}")

# ---- Evaluation / sanity check ----
x_eval = jnp.linspace(0, 1, 200)  # grid for plotting/metrics
u_pred = u_hat(params, x_eval)  # PINN prediction
u_true = jnp.sin(pi * x_eval)  # analytic solution for chosen f

# relative L2 error to gauge quality
l2_rel = jnp.linalg.norm(u_pred - u_true) / jnp.linalg.norm(u_true)
print("Relative L2 error:", float(l2_rel))

# u_pred vs u_true
plt.figure()
plt.plot(x_eval, u_pred, label="PINN")
plt.plot(x_eval, u_true, "--", label="True")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Solution")
plt.legend()
plt.grid(True)

# Residual
uxx = jax.vmap(lambda t: u_xx(params, t))(x_eval)
res = -uxx - f(x_eval)
plt.figure()
plt.plot(x_eval, res)
plt.xlabel("x")
plt.ylabel(r"$-u''_{\theta}(x)-f(x)$")
plt.title("PDE residual")
plt.grid(True)

# 3) Absolute error (only if u_true known)
err = jnp.abs(u_pred - u_true)
plt.figure()
plt.plot(x_eval, err)
plt.xlabel("x")
plt.ylabel("|error|")
plt.title("Absolute error")
plt.grid(True)

# 4) Training curve
plt.figure()
plt.semilogy(loss_hist)
plt.xlabel("iteration")
plt.ylabel("physics loss")
plt.title("Training loss")
plt.grid(True)

plt.show()
