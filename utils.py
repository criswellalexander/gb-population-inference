import jax.numpy as jnp
import numpy as np
from scipy.signal import welch

def get_rfft(data, times, sample_rate):
    return jnp.fft.rfftfreq(data.size, d=1/sample_rate), jnp.fft.rfft(data, norm="backward")

def construct_separate_signals(amps, freqs, phases, times):
    return amps * jnp.sin(2 * np.pi * freqs * times + phases)

def construct_full_signal(amps, freqs, phases, times):
    times = times[:,None]
    signals = construct_separate_signals(amps, freqs, phases, times)
    total_signal = jnp.sum(signals, axis=-1)
    return total_signal

def generate_time_domain_detector_noise(time_array, noise_amplitude, noise_seed=0):
    noise = np.random.randn(time_array.size, ) * noise_amplitude
    return noise

def generate_detector_noise_psd(sample_rate, duration, noise_amplitude):
    deltaT = 1/sample_rate
    times = np.arange(0, duration, deltaT)
    td_noise = generate_time_domain_detector_noise(times, noise_amplitude)
    f, noise_psd = welch(td_noise, fs=sample_rate, nperseg=times.size)
    noise_psd /= 2
    return {'frequencies': f, 'noisePSD': noise_psd}