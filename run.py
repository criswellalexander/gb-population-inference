import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

"""

write functions here to load in the data and the noise psd

"""

fmin =
fmax = 


class Likelihood(object):
    def __init__(self, frequencies, fft_data, detector_noise_psd, duration):
        self.parameters = dict()
        self.fft_data = fft_data
        self.detector_noise_psd = detector_noise_psd
        self.duration = duration
        self.deltaF = frequencies[1] - frequencies[0]
        self.deltaT = 1/self.deltaF
        self.time_array = jnp.arange(0, self.duration, self.deltaT)
        
    def log_likelihood(self):
        resolved_strain = get_resolved_signals(self.parameters, self.time_array)
        unresolved_psd = get_unresolved_psd(self.parameters, self.frequencies, self.detector_noise_psd, self.Nres)
        total_psd = detector_noise_psd + unresolved_psd
        return -jnp.sum(2 * jnp.abs(resolved_strain - fft_data)**2 / (self.duration * total_psd)) - jnp.sum(jnp.log(np.pi * total_psd * self.duration))


def run_model(frequencies, fft_data, detector_noise_psd, duration, Nres):
    frequencies, fft_data, detector_noise_psd, duration = map(jnp.array, (frequencies, fft_data, detector_noise_psd, duration))
    
    likelihood = Likelihood(frequencies, fft_data, detector_noise_psd, duration)
    
    sample = dict()
    
    sample['alpha'] = numpyro.sample('alpha', dist.Uniform(-5, 5))
    sample['beta'] = numpyro.sample('beta', dist.Uniform(-5, 5))
    
    frequencies = jnp.zeros(Nres)
    phases = jnp.zeros(Nres)
    amplitudes = jnp.zeros(Nres)
    
    fmin_here = fmin
    for i in range(Nres):
        frequencies[i] = numpyro.sample(f'frequency_{i}', dist.Uniform(fmin_here, fmax))
        fmin_here = frequencies[i]
        amplitudes[i] = numpyro.sample(f'amplitude_{i}',  dist.Uniform(ampmin, ampmax))
        phases[i] = numpyro.sample(f'phase_{i}',  dist.Uniform(0, 2*np.pi))
        
    sample['frequencies'] = frequencies
    sample['phases'] = phases
    sample['amplitudes'] = amplitudes
    
    likelihood.parameters.update(sample)
    
    loglike = numpyro.factor('loglike', likelihood.log_likelihood())