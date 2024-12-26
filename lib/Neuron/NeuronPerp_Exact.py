import jax
import jax.numpy as jnp
from functools import partial, wraps
from jax.scipy.special import erfinv

from .neuron_base import Neuron_Base

class Neuron_Int(Neuron_Base):
    """
    ndR is the number of dendrites updated at each step, they are selected based on v=w*x/|w|
    weight update Δw exactly solves the following equations:
    (w+Δw)*x = y*|x| = b+kappa
    |w+Δw|^2 = 1

    The output is Σ ReLU(w*x-b)
    """

    def _params_init(self, bias, kappa, key):
        params = super()._params_init(key)
        params["b"] = bias
        params["kappa"] = kappa
        return params

    @partial(jax.jit, static_argnums=(0, ))
    def update_fun(self, params, x):
        x = jnp.atleast_2d(x)
        w, b, kappa = params["w"], params["b"], params["kappa"]
        w_lens = jnp.linalg.norm(w, axis=-1, ord=2) # dim=(nd, )
        wx = jnp.sum(w*x, axis=-1) # dim = (nd, )
        v = wx/w_lens # v = w*x/|w| ~ N(0, 1), this is what the selection is based on
        v_u, idx_u = jax.lax.top_k(v, self.ndR) # the updated ones
            
        x_u, w_u = x[idx_u], w[idx_u] # dim=(ndR, ns)
        w_lens_u, us_u = w_lens[idx_u], wx[idx_u] # dim=(ndR, )
        x_len_u = jnp.linalg.norm(x_u, axis=-1, ord=2) # dim=(ndR, )
        v_over_x_u = v_u/x_len_u # dim=(ndR, )

        y = (b+kappa)/x_len_u # dim=(ndR, ), after learning w*x = y*|x| = b+kappa
        
        tmp1 = jnp.sqrt((1-y**2)/(1-v_over_x_u**2)) # dim = (ndR, )
        tmp2 = ((y-v_over_x_u*tmp1)/x_len_u).reshape((-1, 1))*x_u # dim=(ndR, ns)
        tmp3 = (tmp1/w_lens_u-1).reshape((-1, 1))*w_u # dim=(ndR, ns)
        
        updated_w = w_u + jnp.heaviside(b+kappa-us_u, 0.).reshape((-1, 1))*(tmp2+tmp3)
        w = w.at[idx_u].set(updated_w)
        params["w"] = w
        return params

class Neuron_NonInt_1(Neuron_Base):
    """
    On average ndR is the number of dendrites updated at each step, they are selected based on v=w*x/|w|
    weight update Δw exactly solves the following equations:
    (w+Δw)*x = y*|x| = b+kappa
    |w+Δw|^2 = 1

    The output is Σ ReLU(w*x-b)
    """

    def _params_init(self, bias, kappa, key):
        params = super()._params_init(key)
        params["b"] = bias
        params["kappa"] = kappa
        params["la"] = bias - jnp.sqrt(2.)*erfinv(1-2*self.ndR/self.nd)
        return params

    @partial(jax.jit, static_argnums=(0, ))
    def update_fun(self, params, x):
        x = jnp.atleast_2d(x)
        w, b, kappa, la = params["w"], params["b"], params["kappa"], params["la"]
        w_lens = jnp.linalg.norm(w, axis=-1, ord=2) # dim=(nd, )
        x_lens = jnp.linalg.norm(x, axis=-1, ord=2) # dim=(nd, )
        wx = jnp.sum(w*x, axis=-1) # dim = (nd, )
        v = wx/w_lens # v = w*x/|w| ~ N(0, 1), this is what the selection is based on
        mask1 = jnp.heaviside(v-b+la, 0.) # dim = (nd, )
        mask2 = jnp.heaviside(kappa+b-v, 0.) # dim = (nd, )
             
        v_over_x = v/x_lens # dim=(nd, )
        y = (b+kappa)/x_lens # dim=(nd, ), after learning w*x = y*|x| = b+kappa
        y = jnp.where(y<=1, y, 1.) # y is the cosine value between x and the updated w, which should not exceed 1.
        
        tmp1 = jnp.sqrt((1-y**2)/(1-v_over_x**2)) # dim = (nd, )
        tmp2 = ((y-v_over_x*tmp1)/x_lens).reshape((-1, 1))*x # dim=(nd, ns)
        tmp3 = (tmp1/w_lens-1).reshape((-1, 1))*w # dim=(nd, ns)
        
        updated_w = w + (mask1*mask2).reshape((-1, 1))*(tmp2+tmp3)
        params["w"] = updated_w
        return params

class Neuron_NonInt_FixedMargin(Neuron_Base):
    """
    On average ndR is the number of dendrites updated at each step, they are selected based on v=w*x/|w|
    weight update Δw exactly solves the following equations:
    (w+Δw)*x = y*|x| = = same among the selected dendrite. y is chosen so that the output right after learning is always n*kappa
    |w+Δw|^2 = 1

    The output is Σ ReLU(w*x-b)
    """

    def _params_init(self, bias, kappa, key):
        params = super()._params_init(key)
        params["b"] = bias
        params["kappa"] = kappa
        params["la"] = bias - jnp.sqrt(2.)*erfinv(1-2*self.ndR/self.nd)
        return params

    @partial(jax.jit, static_argnums=(0, ))
    def update_fun(self, params, x):
        x = jnp.atleast_2d(x)
        w, b, kappa, la = params["w"], params["b"], params["kappa"], params["la"]
        w_lens = jnp.linalg.norm(w, axis=-1, ord=2) # dim=(nd, )
        x_lens = jnp.linalg.norm(x, axis=-1, ord=2) # dim=(nd, )
        wx = jnp.sum(w*x, axis=-1) # dim = (nd, )
        v = wx/w_lens # v = w*x/|w| ~ N(0, 1), this is what the selection is based on
        mask1 = jnp.heaviside(v-b+la, 0.) # dim = (nd, )
        mask2 = jnp.heaviside(kappa+b-v, 0.) # dim = (nd, )
        n1, n2 = jnp.sum(mask1), self.nd-jnp.sum(mask2)
        n1 = jnp.max(jnp.array([1, n1]))
            
        factor = (self.ndR-n2)/jnp.max(jnp.array([1, n1-n2]))    
        v_over_x = v/x_lens # dim=(nd, )
        y = (b+factor*kappa)/x_lens # dim=(nd, ), after learning w*x = y*|x| = b+kappa*(ndR-n2)/(n1-n2)
        y = jnp.where(y<=1, y, 1.) # y is the cosine value between x and the updated w, which should not exceed 1. 
        
        tmp1 = jnp.sqrt((1-y**2)/(1-v_over_x**2)) # dim = (nd, )
        tmp2 = ((y-v_over_x*tmp1)/x_lens).reshape((-1, 1))*x # dim=(nd, ns)
        tmp3 = (tmp1/w_lens-1).reshape((-1, 1))*w # dim=(nd, ns)
        
        updated_w = w + (mask1*mask2).reshape((-1, 1))*(tmp2+tmp3)
        params["w"] = updated_w
        return params
    