import jax
import jax.numpy as jnp
import numpy as np
from functools import partial, wraps
from tqdm import tqdm

from .. import Neuron
from .. import XsGenerator

class Simulation_Run():
    def __init__(self, neuron: Neuron.neuron_base.Neuron_Base, xs_gen: XsGenerator.xs_generator_base.Xs_Generator_Base, params, decay_steps=500, initial_steps=500, n_tested_patterns=100, refresh_every=1000, seed=42) -> None:
        self.key = jax.random.PRNGKey(seed)
        self.key, subkey1, subkey2 = jax.random.split(self.key, 3)
        self.neuron = neuron
        self.params = params
        self.decay_steps = decay_steps
        self.initial_steps = initial_steps
        self.n_test_patterns = n_tested_patterns
        self.refresh_every = max(refresh_every, n_tested_patterns)
        self.xs_gen = xs_gen
        self.xs = self.xs_gen.gen(subkey1, self.refresh_every)
        self.params = self.init_w(subkey2, self.params)

        self.outputs_record = np.zeros((n_tested_patterns, decay_steps+n_tested_patterns))

    @partial(jax.jit, static_argnums=(0, ))
    def _neuron_update_fun(self, params, x):
        return self.neuron.update_fun(params, x)
    
    @partial(jax.jit, static_argnums=(0, ))
    def _neuron_get_output(self, params, x0s):
        return self.neuron.get_output(params, x0s)

    def init_w(self, key, params):
        for i  in range(self.initial_steps):
            if i%self.refresh_every == 0:
                subkey, key = jax.random.split(key)
                xs0 = self.xs_gen.gen(subkey, self.refresh_every)
            params = self._neuron_update_fun(params, xs0[i%self.refresh_every])
        return params

    @partial(jax.jit, static_argnums=(0, ))
    def _update_and_get_outputs(self, params, x, x0s):
        params = self._neuron_update_fun(params, x)
        votes = jax.vmap(self._neuron_get_output, in_axes=(None, 0))(params, x0s) # dim=(n_tested_patterns, )
        return params, votes

    def run(self, progress_bar=True):
        self.key, subkey = jax.random.split(self.key, 2)
        xs = self.xs_gen.gen(subkey, self.refresh_every)           
        x0s = jnp.copy(xs[:self.n_test_patterns]) # in jax numpy, a copy is created for x0s
        if progress_bar is True: 
            pb = tqdm(range(self.decay_steps+self.n_test_patterns))
        else:
            pb = range(self.decay_steps+self.n_test_patterns)
        for i in pb:
            if i%self.refresh_every == 0 and i>=self.refresh_every:
                self.key, subkey = jax.random.split(self.key)
                xs = self.xs_gen.gen(subkey, self.refresh_every)
            self.params, votes = self._update_and_get_outputs(self.params, xs[i%self.refresh_every], x0s)
            self.outputs_record[..., i] = votes
        
        for i in range(self.n_test_patterns):
            self.outputs_record[i] = np.roll(self.outputs_record[i], -i, axis=-1)