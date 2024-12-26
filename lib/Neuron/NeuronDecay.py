import jax
import jax.numpy as jnp
from functools import partial, wraps
from jax.scipy.special import erfinv

from .neuron_base import Neuron_Base

class Neuron_NonInt(Neuron_Base):
    """update rule is Δw = x/|x|^2 (b+k-(1-beta/ns)*w*x) - beta/ns w, on average ndR dendrites are updated each time. """

    def _params_init(self, bias, kappa, beta, key):
        params = super()._params_init(key)
        params["b"] = bias
        params["kappa"] = kappa
        params["beta"] = beta
        params["la"] = bias - jnp.sqrt(2.)*erfinv(1-2*self.ndR/self.nd)
        return params

    @partial(jax.jit, static_argnums=(0, ))
    def update_fun(self, params, x):
        x = jnp.atleast_2d(x) # dim = (nd, ns) or (1, ns)
        w, b, kappa, beta, la = params["w"], params["b"], params["kappa"], params["beta"], params["la"]
        x_len2 = jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True)**2
        overlaps = jnp.sum(w*x, axis=-1) # dim = (nd, )
        mask = jnp.heaviside(kappa+b-overlaps, 0)*jnp.heaviside(overlaps-b+la, 0.) # dim = (nd, )
        w = w + mask.reshape((-1, 1))*(x/x_len2*(b+kappa-(1-beta/self.ns)*overlaps).reshape((-1, 1)) - beta/self.ns*w) # dim = (ndR, ns)
        params["w"] = w
        return params
    
class Neuron_NonInt_FixedMargin(Neuron_Base):
    """update rule is Δw = x/|x|^2 (y-(1-beta/ns)*w*x) - beta/ns w, on average ndR dendrites are updated each time. """

    def _params_init(self, bias, kappa, beta, key):
        params = super()._params_init(key)
        params["b"] = bias
        params["kappa"] = kappa
        params["beta"] = beta
        params["la"] = bias - jnp.sqrt(2.)*erfinv(1-2*self.ndR/self.nd)
        return params

    @partial(jax.jit, static_argnums=(0, ))
    def update_fun(self, params, x):
        x = jnp.atleast_2d(x) # dim = (nd, ns) or (1, ns)
        w, b, kappa, beta, la = params["w"], params["b"], params["kappa"], params["beta"], params["la"]
        x_len2 = jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True)**2
        overlaps = jnp.sum(w*x, axis=-1) # dim = (nd, )
        mask1 = jnp.heaviside(overlaps-b+la, 0.) # dim = (nd, )
        mask2 = jnp.heaviside(kappa+b-overlaps, 0) # dim = (nd, )
        n1, n2 = jnp.sum(mask1), jnp.sum(mask2)
        w = w + (mask1*mask2).reshape((-1, 1))*(x/x_len2*(b+((self.ndR-n2)/n1)*kappa-(1-beta/self.ns)*overlaps).reshape((-1, 1)) - beta/self.ns*w) # dim = (ndR, ns)
        params["w"] = w
        return params
