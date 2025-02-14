import jax.numpy as jnp
import numpy as np
from scipy.signal.windows import hann
from scipy.signal import welch
from gwpy.timeseries import TimeSeries

def get_resolved_signals_td(parameters, time_array, sample_rate):
    time_domain_signal = construct_full_signal(parameters['amplitudes'], parameters['frequencies'], parameters['phases'], time_array)
    return time_domain_signal

def get_resolved_signals(parameters, time_array, sample_rate):
    time_domain_signal = construct_full_signal(parameters['amplitudes'], parameters['frequencies'], parameters['phases'], time_array)
    _, frequency_domain_signal = get_rfft(time_domain_signal, time_array, sample_rate)
    return frequency_domain_signal

def get_rfft(data, times, sample_rate):
    fft = jnp.fft.rfft(data) # / data.size
    freqs = jnp.fft.rfftfreq(data.size, d=1/sample_rate)
    return freqs[1:], fft[1:] / sample_rate

def construct_separate_signals(amps, freqs, phases, times):
    return amps * jnp.sin(2 * np.pi * freqs * times + phases)

def construct_full_signal(amps, freqs, phases, times):
    times = times[:,None]
    signals = construct_separate_signals(amps, freqs, phases, times)
    total_signal = jnp.sum(signals, axis=-1)
    return total_signal

def generate_time_domain_detector_noise(time_array, noise_amplitude):
    noise = np.random.randn(time_array.size) * noise_amplitude
    return noise

def generate_detector_noise_psd(sample_rate, duration, noise_amplitude):
    deltaT = 1/sample_rate
    times = np.arange(0, duration, deltaT)
    td_noise = generate_time_domain_detector_noise(times, noise_amplitude)
    f, noise_psd = welch(td_noise, fs=sample_rate, nperseg=times.size)
    noise_psd /= 2
    # psd = TimeSeries(td_noise, times=times).psd()
    return {'frequencies': f, 'noisePSD': noise_psd}



### Seeded versions of some of the above:
def generate_time_domain_detector_noise_with_seed(time_array, noise_amplitude, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=time_array.size) * noise_amplitude
    return noise

def generate_detector_noise_psd_with_seed(sample_rate, duration, noise_amplitude, seed=0):
    deltaT = 1/sample_rate
    times = np.arange(0, duration, deltaT)
    td_noise = generate_time_domain_detector_noise_with_seed(times, noise_amplitude, seed=seed)
    f, noise_psd = welch(td_noise, fs=sample_rate, nperseg=times.size)
    noise_psd /= 2
    # psd = TimeSeries(td_noise, times=times).psd()
    return {'frequencies': f, 'noisePSD': noise_psd}
