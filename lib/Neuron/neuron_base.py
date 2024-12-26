import jax
import jax.numpy as jnp
from functools import partial, wraps

class Neuron_Base():
    """relu function is the default activation function. 
    params contains {"w", "b"}"""

    def __init__(self, n_synapses, n_dendrites, ndR) -> None:
        self.ns = n_synapses # number of synapses per dendrite. 
        self.nd = n_dendrites
        self.ndR = ndR

    def _params_init(self, key):
        w = jax.random.normal(key, (self.nd, self.ns))*1.0/jnp.sqrt(self.ns)
        params = {"w": w, "b":  0.}
        return params

    @wraps(partial(jax.jit, static_argnums=(0, )))
    def update_fun(self, params, x):
        return params
    
    @wraps(partial(jax.jit, static_argnums=(0, )))
    def get_output(self, params, x):
        "get the neuron's output to decide whether pattern x is familiar or unfamiliar"
        w = params["w"]
        b = params["b"]
        x = jnp.atleast_2d(x)
        us = jnp.sum(w*x, axis=-1) # dim=(nd, )
        output = jnp.sum(jax.nn.relu(us - b)) # dim=0
        return output