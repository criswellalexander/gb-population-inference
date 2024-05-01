from dataclasses import dataclass, field
from typing import List, Union, Dict
import numpy as np
import scipy
from .distributions import *
import pandas as pd
import matplotlib.pyplot as plt
#import numpyro.distributions as dist

class AbstractInjectionPopulationModel:
	pass


DEFAULT_RANGES = {'chirp_mass' : [0.5, 1.4], 
				  'seperation': [0.5, 20], 
				  'luminosity_distance':[1, 50], 
				  'phase':[0,2*np.pi]}

@dataclass
class PowerLawChirpPowerLawSeperation:
	limits : Dict[str, List]

	def __post_init__(self):

		self.A_scale = 1.385328279341387e-20
		self.f_scale = 0.011520088541326409

		self.A_i = lambda Mc, d: A_scale * (Mc**(5/3) / d)  # A_i(Mc, d)
		self.A_i_f = lambda Mc, d, f: A_scale * (Mc**(5/3) / d) * (np.pi * f)**(2/3)  # A_i(Mc, d, f)
		self.f_i = lambda Mc, r: f_scale * np.sqrt(Mc / np.pi) * r**(-3/2)  # f_i(Mc, r)
		self.timeseries = lambda t, f, phi, A: A * np.sin((2 * np.pi * f * t) + phi)


		def create_distribution(Lambda)
		 	return dict(Mc = TruncatedPowerLaw(Lambda['alpha'], *limits["chirp_mass"]),
                        r = TruncatedPowerLaw(Lambda['beta'], *limits["seperation"]),
                        d = TruncatedPowerLaw(1.0, *limits["luminosity_distance"]),
                        phi = Uniform(*limits['phase']))

		 self.distribution_func = self.create_distribution


	def generate_samples(Lambda, size=1):
		return {key : dist.sample(size) for key,dist in self.distribution_func.items()}

	def generate_time_series(Lambda, sample_rate=0.25, duration=10000):
		PopulationInjection = self.generate_samples(Lambda)
		Mc, r, d, phi = PopulationInjection['Mc'], PopulationInjection['r'], PopulationInjection['d'], PopulationInjection['phi']
		ts = np.linspace(0, duration, int(sample_rate * duration))
		return ts, self.timeseries(ts[:, None], self.f_i(Mc, r), phi, self.A_i(Mc, d)).sum(axis=-1)

	def plot_time_series(ts, Ys):
		fig, ax = plt.subplots(dpi=150)
		plt.plot(ts, Ys)
		plt.xlabel(r"$t (s)$")
		plt.ylabel(r"strain")
		plt.title(f"Strain for population parameters alpha = {np.round(Lambda['alpha'],1)}, beta = {np.round(Lambda['beta'],1)}")
		plt.grid(alpha=0.3)
		plt.show()


