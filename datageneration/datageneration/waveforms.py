from dataclasses import dataclass, field
from typing import List, Union, Dict, Any
import numpy as np
import scipy
from .distributions import *
import pandas as pd
import matplotlib.pyplot as plt

class AbstractWaveform:
	pass

get_function_args = lambda fn : fn.__code__.co_varnames[:fn.__code__.co_argcount]

@dataclass
class SinosoidWaveform(AbstractWaveform):
	amplitude_model : Any = lambda amplitude : amplitude
	frequency_model : Any = lambda frequency : frequency
	phase_model : Any = lambda phase : phase 

	def compute_waveform_parameters(self, parameters, in_place=True):
		amplitude_args = {col : parameters[col] for col in parameters.keys() if col in get_function_args(self.amplitude_model)}
		frequency_args = {col : parameters[col] for col in parameters.keys() if col in get_function_args(self.frequency_model)}
		phase_args = {col : parameters[col] for col in parameters.keys() if col in get_function_args(self.phase_model)}
		A = self.amplitude_model(**amplitude_args)
		f = self.frequency_model(**frequency_args)
		phase = self.phase_model(**phase_args)
		result = {'amplitude' : A, 'frequency' : f, 'phase' : phase, **parameters}
		if in_place:
			#result.update(parameters)
			return result
		else:
			return result

	def generate_waveforms(self, parameters : Dict[str, Union[float, np.array]], sample_rate=0.25, duration=10000):
		waveformparams = self.compute_waveform_parameters(parameters, in_place=True)

		A = waveformparams['amplitude']
		f = waveformparams['frequency']
		phase =waveformparams['phase']

		ts = np.linspace(0, duration, int(sample_rate * duration))

		if all(isinstance(variable, float) for variable in (A,f,phase)):
			self.timeseries = A * np.sin((2 * np.pi * f * ts) + phase)
		else:
			self.timeseries = A * np.sin((2 * np.pi * f * ts[:,None]) + phase)

		return ts, self.timeseries

