import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from utils import get_resolved_signals

"""

write functions here to load in the data and the noise psd

"""

class Likelihood(object):
    def __init__(self, frequencies, fft_data, detector_noise_psd, duration):
        self.parameters = dict()
        self.fft_data = fft_data
        self.detector_noise_psd = detector_noise_psd
        self.duration = duration
        self.deltaF = frequencies[1] - frequencies[0]
        self.deltaT = 1/self.deltaF
        self.time_array = jnp.arange(0, self.duration, self.deltaT)
        self.frequencies = frequencies
        
    def log_likelihood(self):
        resolved_strain = get_resolved_signals(self.parameters, self.time_array, self.frequencies[-1]*2)
        unresolved_psd = get_unresolved_psd(self.parameters, self.frequencies, self.detector_noise_psd, self.Nres)
        total_psd = self.detector_noise_psd + unresolved_psd
        return -jnp.sum(2 * jnp.abs(resolved_strain - self.fft_data)**2 / (self.duration * total_psd)) - jnp.sum(jnp.log(np.pi * total_psd * self.duration))


def run_model(likelihood, Nres, ampmin=0, ampmax=10):

    sample = dict()
    
    sample['alpha'] = numpyro.sample('alpha', dist.Uniform(-5, 5))
    sample['beta'] = numpyro.sample('beta', dist.Uniform(-5, 5))
    
    resolved_frequencies = jnp.zeros(Nres)
    phases = jnp.zeros(Nres)
    amplitudes = jnp.zeros(Nres)
    
    fmin_here = likelihood.frequencies[0]
    fmax = likelihood.frequencies[-1]
    
    for i in range(Nres):
        resolved_frequencies.at[i].set(numpyro.sample(f'frequency_{i}', dist.Uniform(fmin_here, fmax)))
        fmin_here = resolved_frequencies[i]
        amplitudes.at[i].set(numpyro.sample(f'amplitude_{i}',  dist.Uniform(ampmin, ampmax)))
        phases.at[i].set(numpyro.sample(f'phase_{i}',  dist.Uniform(0, 2*np.pi)))
        
    sample['frequencies'] = resolved_frequencies
    sample['phases'] = phases
    sample['amplitudes'] = amplitudes
    sample['N_total'] = numpyro.sample('N_total', dist.Uniform(Nres * 100, Nres * 10000))
    
    likelihood.parameters.update(sample)
    
    loglike = numpyro.factor('loglike', likelihood.log_likelihood())