import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
import matplotlib.pyplot as plt
from functools import partial

m = 1.0
mu = 0.4
k = 4.0


def exact_solution(t):
    """
    Function that calculates the exact solution of the ODE m * u_tt + mu * u_t + k * u = 0

    Parameters:
        t -> Array: Array of timepoints
    Returns:
        Exact solution to ODE
    """
    w = jnp.sqrt(4 * k - mu**2) / 2
    return jnp.exp(-mu / 2 * t) * jnp.cos(w * t)


class NeuralNet(nn.Module):
    hidden_dim: int = 32  # Hidden layers dimension
    n_hidden: int = 2  # Number of hidden layers

    @nn.compact  # Covers everything an __init__ would.
    def __call__(self, t):
        """
        Function that puts data through neural network.

        Parameters:
            t -> Array: Array of timepoints
        returns:
            x -> scalar: Value predicted by NeuralNet
        """
        if t.ndim == 0:
            t = t[None, None]
        elif t.ndim == 1:
            t = t[:, None]
        x = t

        # For each hidden layer, put data through activation function and update x for next hidden layer.
        for _ in range(self.n_hidden):
            x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        return (
            nn.Dense(1)(x).squeeze()
        )  # Returns x after going through output layer (This is our predicted value).


def residual(params, model, t):
    """
    Fucntion that computes the residual of the damped oscillator at given time t.

    Parameters:
        params -> Dict: Neural network paramters from flax __init__.

        model -> flax.linen.module: Neural network object that approximates u(t).

        t -> array or scalar: single timepoint on which to calculate residual.

    Returns:
        float: Value of ODE residual at time t
    """

    def u_fn(t_scalar):
        """
        Helper function that puts one timepoint through neural network

        Parameters:
            t_scalar -> float: Single time point
        returns:
            out -> float: Neural network prediction for specified timepoint
        """
        t_scalar = jnp.atleast_1d(t_scalar)
        out = model.apply(params, t_scalar.reshape(1, 1)).squeeze()
        return out

    # calculate gradients
    du_fn = jax.grad(
        u_fn
    )  # returns function with same parameters as u_fn --> 1st derivative
    ddu_fn = jax.grad(
        du_fn
    )  # returns function with same parameters as u_fn --> 1st derivative

    u = u_fn(t)
    du = du_fn(t)
    ddu = ddu_fn(t)

    # return ODE using neural network predictions and gradient calculations:
    return m * ddu + mu * du + k * u


@partial(
    jax.jit, static_argnums=(1,)
)  # Treat argument at index 1 as static (Model does not change during computation)
def loss_fn(params, model, ts):
    """
    Function that calculates total loss

    Parameters:
        params -> Dict: Neural network paramters from flax __init__.
        model --> flax.linen.module: Neuralk network object that approximates u(t)
        ts --> Array: Array of all timepoints
    Returns:
        total_loss --> float: Total loss
    """
    res = jax.vmap(lambda t: residual(params, model, t))(
        ts.flatten()
    )  # Applies residual function to all ts in one go (vecotrized)
    res_loss = jnp.mean(
        res**2
    )  # Calculate loss - measure of how well the NN fits the ODE

    u0 = (
        model.apply(params, jnp.array([[0.0]])).squeeze()
    )  # Apply model to initial timepoint (models prediction at first timepoint)
    du_dt_fn = jax.grad(lambda t: model.apply(params, jnp.array([[t]])).squeeze())
    du0 = du_dt_fn(0.0)

    ic_loss = (u0 - 1.0) ** 2 + (du0 - 0.0) ** 2
    total_loss = res_loss + ic_loss

    return total_loss


def train(model, ts, num_steps=5000, lr=1e-2):
    """
    Model that train Neural Network on given timesteps

    Parameters:
        model --> flax.linen.module: Neuralk network object that approximates u(t)
        ts --> Array: Array of all timesteps
        num_steps --> int: Number of training iterations (default 5000)
        lr --> float: Learning rate for optimizer
    Returns:
        params -> Dict: Trained neural network parameters.
    """
    rng = jax.random.PRNGKey(0)
    dummy_input = ts[:1].reshape(1, 1)  # Used to initialize model params
    params = model.init(rng, dummy_input)  # initialize params

    optimizer = optax.adam(lr)  # ADAM optimizer with learning rate lr
    opt_state = optimizer.init(
        params
    )  # Initial optimizer state (to keep track of moments)

    @jax.jit
    def step(params, opt_state):
        """
        Function that performs one training step

        Parameters:
            params -> Dict: Trained neural network parameters.
            opt_state
        Returns:
            params -> Dict: Trained neural network parameters.
            opt_state -> Dict: Dictionary of optimizer information at current state
            loss --> float: Loss at current step
        """
        loss, grads = jax.value_and_grad(loss_fn)(params, model, ts)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    for i in range(num_steps):
        params, opt_state, loss = step(params, opt_state)
        if i % 500 == 0:
            print(f"Step {i}, Loss: {loss:.4e}")  # print loss for every 500 steps

    return params


ts_train = jnp.linspace(0, 10, 100).reshape(-1, 1)
model = NeuralNet()
trained_params = train(model, ts_train)

ts_plot = jnp.linspace(0, 10, 200).reshape(-1, 1)
u_pred = jax.vmap(lambda t: model.apply(trained_params, t))(ts_plot)
u_exact = exact_solution(ts_plot)

plt.plot(ts_plot, u_pred, label="PINN")
plt.plot(ts_plot, u_exact, label="Exact", linestyle="dashed")
plt.xlabel("t")
plt.ylabel("u(t)")
plt.legend()
plt.title("Damped Spring ODE: PINN vs Exact")
plt.savefig("figs/PINN_VS_Exact.pdf")
