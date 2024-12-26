import jax
import jax.numpy as jnp
from functools import partial, wraps

class Xs_Generator_Base():
    def __init__(self, nd, ns, seed=0, *args, **kwargs) -> None:
        self.key = jax.random.PRNGKey(seed)
        self.ns = ns
        self.nd = nd

    @wraps(partial(jax.jit, static_argnums=(0, 2)))
    def gen(self, key, n_patterns) -> jnp.ndarray:
        return jnp.zeros((n_patterns, self.nd, self.ns))