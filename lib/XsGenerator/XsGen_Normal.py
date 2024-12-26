import jax
import jax.numpy as jnp
from functools import partial, wraps

from .xs_generator_base import Xs_Generator_Base

class XSGen_Normal(Xs_Generator_Base):
    """generate input patterns, each element of the pattern is sampled from a standard normal distribution. 
    For the same input case, set nd=1"""
    def __init__(self, nd, ns, normalized_len, seed=0, *args, **kwargs) -> None:
        super().__init__(nd, ns, seed, *args, **kwargs)
        self.normalized_len = normalized_len 

    @wraps(partial(jax.jit, static_argnums=(0, 2)))
    def gen(self, key, n_patterns) -> jnp.ndarray:
        xs = jax.random.normal(key, (n_patterns, self.nd, self.ns))
        def normalize_fun(xs):
            return xs/jnp.linalg.norm(xs, ord=2, axis=-1, keepdims=True)*self.normalized_len
        xs = jax.lax.cond(self.normalized_len>0, normalize_fun, lambda x: x, xs)
        return xs
    
    def get_repeat_num_mat(self) -> jnp.ndarray:
        return jnp.ones((self.nd, self.ns))
    
