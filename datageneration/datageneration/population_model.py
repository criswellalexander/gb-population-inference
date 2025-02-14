from dataclasses import dataclass, field
from typing import List, Union, Dict, Any
import numpy as np
import scipy
from .dataloader import write_nested_to_hdf5
from .distributions import *
from .waveforms import *
import pandas as pd
import matplotlib.pyplot as plt



DEFAULT_RANGES = {'chirp_mass' : [0.5, 1.4], 
  'seperation': [0.5, 20], 
  'luminosity_distance':[1, 50], 
  'phase':[0,2*np.pi]}

A_scale = 1.385328279341387e-20
f_scale = 0.011520088541326409

f_i = lambda chirp_mass, seperation: f_scale * np.sqrt(chirp_mass / np.pi) * seperation**(-3/2)  				# f_i(chirp_mass, r)
A_i 		= lambda chirp_mass, luminosity_distance: 				A_scale * (chirp_mass**(5/3) / luminosity_distance)  									# A_i(chirp_mass, luminosity_distance)
A_i_f 		= lambda chirp_mass, luminosity_distance, seperation:  	A_scale * (chirp_mass**(5/3) / d) * (np.pi * f_i(chirp_mass, seperation))**(2/3)     	# A_i(chirp_mass, luminosity_distance, f)
timeseries  = lambda t, A, f, phase:             					A * np.sin((2 * np.pi * f * t) + phase)

SineWDSignal = SinosoidWaveform(
        amplitude_model = A_i,
        frequency_model = f_i,
        phase_model = (lambda phase : phase)
)


@dataclass
class PowerLawChirpPowerLawSeperation:
    limits : Dict[str, List] = field(default_factory = lambda : {k:v.copy() for k,v in DEFAULT_RANGES.items()})
    distance_power_law_index : int = 1
    waveform : AbstractWaveform = field(default_factory = lambda : SineWDSignal)
    poisson : bool = False
    sample_rate : float = 0.25
    duration : float = 10000
    N_white_dwarfs : int = 1000


    def __post_init__(self):
        def create_distribution(Lambda : Dict[str, float]):
            return dict(chirp_mass = TruncatedPowerLaw(Lambda['alpha'], *self.limits["chirp_mass"]),
        				seperation = TruncatedPowerLaw(Lambda['beta'], *self.limits["seperation"]),
        						luminosity_distance = TruncatedPowerLaw(self.distance_power_law_index, *self.limits["luminosity_distance"]),
        						phase = Uniform(*self.limits['phase']))
            
        self.distribution_func = create_distribution


    def generate_samples(self, Lambda : Dict[str, float], size=1):
        return {key : dist.sample(size) for key,dist in self.distribution_func(Lambda).items()}

    def generate_time_series(self, Lambda : Dict[str, float], N_white_dwarfs=None, **kwargs):
        N_white_dwarfs = N_white_dwarfs or self.N_white_dwarfs
        sample_rate = self.sample_rate or kwargs.get('sample_rate', 0.25)
        duration = self.duration or kwargs.get('duration', 10000)
        summed = kwargs.get('summed',True)
        if self.poisson:
            N_white_dwarfs = Poisson(N_white_dwarfs).sample()
    
        self.samples_from_population = self.generate_samples(Lambda, size=N_white_dwarfs)
        
        ts, timeseries_out = self.waveform.generate_waveforms(self.samples_from_population, sample_rate=sample_rate, duration=duration,summed=summed)
        
#        strain = waveforms.sum(axis=-1)
#        if summed:
#            timeseries_out = strain
#        else:
#            timeseries_out = waveforms
        
        self.samples_from_population = self.waveform.compute_waveform_parameters(self.samples_from_population, in_place=True)
        if summed:
            self.time, self.strain, self.waveforms = ts, timeseries_out, None
        else:
            strain = timeseries_out.sum(axis=-1)
            self.time, self.strain, self.waveforms = ts, strain, timeseries_out
        self.sample_rate, self.duration, self.N_white_dwarfs = sample_rate, duration, N_white_dwarfs
             
        return ts, timeseries_out
    
    def save_to_file(self, filename, extra_parameters={}):
        data = {'strain' : self.strain, 
                    'time' : self.time, 
                    'injected_population' : pd.DataFrame(self.samples_from_population), 
                    'sample_rate' : self.sample_rate, 
                    'duration': self.duration,
                    'limits' : self.limits,
                    **extra_parameters
                    }
        
        write_nested_to_hdf5(data, filename)
        print(f"Data saved to {filename}")
    
    
    def plot_time_series(self, ts, Ys):
        fig, ax = plt.subplots(dpi=150)
        plt.plot(ts, Ys)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"strain")
        plt.grid(alpha=0.3)
        plt.show()



@dataclass
class WhiteDwarfDistribution:
    distribution_function : Dict[str, Any]
    distance_power_law_index : int = 1
    waveform : AbstractWaveform = field(default_factory = lambda : SineWDSignal)
    poisson : bool = False

    def generate_samples(self, Lambda : Dict[str, float], size=1):
        return {key : dist.sample(size) for key,dist in self.distribution_function(Lambda).items()}
    
    def generate_time_series(self, Lambda : Dict[str, float], N_white_dwarfs=1000, **kwargs):
        if self.poisson:
            N_white_dwarfs = Poisson(N_white_dwarfs).sample()
        self.samples_from_population = self.generate_samples(Lambda, size=N_white_dwarfs)
        ts, waveforms = self.waveform.generate_waveforms(self.samples_from_population, **kwargs)
        return ts, waveforms.sum(axis=-1)
    
    def plot_time_series(self, ts, Ys):
        fig, ax = plt.subplots(dpi=150)
        plt.plot(ts, Ys)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"strain")
        plt.grid(alpha=0.3)
        plt.show()