import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from utils import get_resolved_signals, get_resolved_signals_td
import matplotlib.pyplot as plt
"""

write functions here to load in the data and the noise psd

"""

class Likelihood(object):
    def __init__(self, frequencies, fft_data, detector_noise_psd, times, sample_rate):
        self.parameters = dict()
        self.fft_data = fft_data
        self.detector_noise_psd = detector_noise_psd
        self.duration = times[-1] - times[0]
        self.deltaF = frequencies[1] - frequencies[0]
        self.deltaT = 1/sample_rate
        self.time_array = times
        self.frequencies = frequencies
        self.sample_rate =sample_rate
        
    def log_likelihood(self):
        resolved_strain = get_resolved_signals(self.parameters, self.time_array, self.sample_rate)
        # unresolved_psd = get_unresolved_psd(self.parameters, self.frequencies, self.detector_noise_psd, self.Nres)
        total_psd = self.detector_noise_psd#+ unresolved_psd
        return -jnp.sum(2 * jnp.abs(resolved_strain - self.fft_data)**2 / (self.duration * total_psd)) - jnp.sum(jnp.log(np.pi * total_psd * self.duration))

# for jitting greatness
def create_likelihood(frequencies, fft_data, detector_noise_psd, times, sample_rate, Nres):
    duration = times[-1] - times[0]
    def log_likelihood(params):
        resolved_strain = get_resolved_signals(params, times, sample_rate)
        # unresolved_psd = get_unresolved_psd(params, frequencies, detector_noise_psd, Nres)
        total_psd = detector_noise_psd#+ unresolved_psd
        return -jnp.sum(2 * jnp.abs(resolved_strain - fft_data)**2 / (duration * total_psd)) - jnp.sum(jnp.log(jnp.pi * total_psd * duration))
    log_likelihood.frequencies = frequencies
    log_likelihood.fft_data = fft_data
    log_likelihood.detector_noise_psd = detector_noise_psd
    log_likelihood.times = times
    log_likelihood.sample_rate = sample_rate
    return log_likelihood

class TDLikelihood(object):
    def __init__(self, data, times, wn_error, sample_rate):
        self.parameters = dict()
        self.data = data
        self.times = times
        self.wn_error = wn_error
        self.sample_rate = sample_rate
        self.duration = times[-1] - times[0]
        self.frequencies = jnp.arange(1/self.duration, sample_rate / 2., 1/self.duration)
        
    def log_likelihood(self):
        resolved_strain = get_resolved_signals_td(self.parameters, self.times, self.sample_rate)
        return -0.5 * np.sum((self.data - resolved_strain)**2 / self.wn_error**2)

def run_model_jitted_likelihood(jlogl, Nres, ampmin=1e-4, ampmax=5):
    sample = dict()
    
    sample['alpha'] = numpyro.sample('alpha', dist.Uniform(-5, 5))
    sample['beta'] = numpyro.sample('beta', dist.Uniform(-5, 5))
    
    fmin = jlogl.frequencies[0]
    fmax = jlogl.frequencies[-1]

    amplitudes = numpyro.sample(f'amplitude', dist.Uniform(ampmin, ampmax).expand([Nres]))
    phases = numpyro.sample(f'phase', dist.Uniform(0, 2*np.pi).expand([Nres]))
    resolved_frequencies = numpyro.sample('frequency', dist.Uniform(fmin, fmax).expand([Nres]))         

    sample['frequencies'] = resolved_frequencies
    sample['phases'] = phases
    sample['amplitudes'] = amplitudes
    sample['N_total'] = numpyro.sample('N_total', dist.Uniform(Nres * 100, Nres * 10000))
    loglike = numpyro.factor('loglike', jlogl(sample))

def run_model_no_ordering(likelihood, Nres, ampmin=1e-4, ampmax=5):

    sample = dict()
    
    sample['alpha'] = numpyro.sample('alpha', dist.Uniform(-5, 5))
    sample['beta'] = numpyro.sample('beta', dist.Uniform(-5, 5))
    
    fmin_here = likelihood.frequencies[0]
    fmax = likelihood.frequencies[-1]

    amplitudes = numpyro.sample(f'amplitude', dist.Uniform(ampmin, ampmax).expand([Nres]))
    phases = numpyro.sample(f'phase', dist.Uniform(0, 2*np.pi).expand([Nres]))
    resolved_frequencies = numpyro.sample('frequency', dist.Uniform(fmin_here, fmax).expand([Nres]))         

    sample['frequencies'] = resolved_frequencies
    sample['phases'] = phases
    sample['amplitudes'] = amplitudes
    sample['N_total'] = numpyro.sample('N_total', dist.Uniform(Nres * 100, Nres * 10000))
    
    likelihood.parameters.update(sample)
    loglike = numpyro.factor('loglike', likelihood.log_likelihood())

def run_model(likelihood, Nres, ampmin=1e-4, ampmax=3):

    sample = dict()
    
    sample['alpha'] = numpyro.sample('alpha', dist.Uniform(-5, 5))
    sample['beta'] = numpyro.sample('beta', dist.Uniform(-5, 5))

    resolved_frequencies = jnp.zeros(Nres)
    phases = jnp.zeros(Nres)
    amplitudes = jnp.zeros(Nres)
    
    fmin_here = likelihood.frequencies[0]
    fmax = likelihood.frequencies[-1]
    frequencies = numpyro.sample
    
    for i in range(Nres):
        resolved_frequencies = resolved_frequencies.at[i].set(numpyro.sample(f'frequency_{i}', dist.Uniform(fmin_here, fmax)))
        fmin_here = resolved_frequencies[i]
        amplitudes = amplitudes.at[i].set(numpyro.sample(f'amplitude_{i}',  dist.Uniform(ampmin, ampmax)))
        phases = phases.at[i].set(numpyro.sample(f'phase_{i}',  dist.Uniform(0, 2*np.pi)))

    sample['frequencies'] = resolved_frequencies
    sample['phases'] = phases
    sample['amplitudes'] = amplitudes
    sample['N_total'] = numpyro.sample('N_total', dist.Uniform(Nres * 100, Nres * 10000))
    
    likelihood.parameters.update(sample)
    loglike = numpyro.factor('loglike', likelihood.log_likelihood())