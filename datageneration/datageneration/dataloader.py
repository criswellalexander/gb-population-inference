from dataclasses import dataclass
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

def write_nested_to_hdf5(data, filename):
    def write_group(group, data):
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                write_group(subgroup, value)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, pd.DataFrame):
                subgroup = group.create_group(key)
                for col_name, col_data in value.items():
                    subgroup.create_dataset(col_name, data=col_data)
            elif isinstance(value, pd.Series):
                value = pd.DataFrame({col : [val] for col,val in value.to_dict().items()})
                subgroup = group.create_group(key)
                for col_name, col_data in value.items():
                    subgroup.create_dataset(col_name, data=col_data)
            elif isinstance(value, float):
                group.attrs[key] = value
            elif isinstance(value, int):
                group.attrs[key] = value

    with h5py.File(filename, 'w') as f:
        write_group(f, data)
        
        
def read_hdf5_to_dict(filename):
    data = {}
    with h5py.File(filename, 'r') as f:
        for key in f.attrs.keys():
            data[key] = f.attrs[key]
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                subgroup = f[key]
                subgroup_data = {}
                for subkey in subgroup.keys():
                    if isinstance(subgroup[subkey], h5py.Dataset):
                        subgroup_data[subkey] = np.array(subgroup[subkey])
                    elif isinstance(subgroup[subkey], h5py.Group):
                        subsubgroup = subgroup[subkey]
                        subsubgroup_data = {}
                        for subsubkey in subsubgroup.keys():
                            subsubgroup_data[subsubkey] = np.array(subsubgroup[subsubkey])
                        subgroup_data[subkey] = subsubgroup_data
                data[key] = subgroup_data
            elif isinstance(f[key], h5py.Dataset):
                data[key] = np.array(f[key])
            else:
                data[key] = f.attrs[key]

    return data

def calculate_psd(strain, time, fs):
    f, psd = welch(strain, fs=fs, nperseg=len(strain))
    return f, psd

@dataclass
class DataLoader:
    filename : str
    
    def __post_init__(self):
        self.loaded_dict = read_hdf5_to_dict(self.filename)
        self.loaded_dict['injected_population'] = pd.DataFrame(self.loaded_dict['injected_population'])
        self.loaded_dict['population_parameters'] = pd.DataFrame(self.loaded_dict['population_parameters'])
        self.__dict__.update(self.loaded_dict)
        self.strain = self.loaded_dict['strain']
        self.time = self.loaded_dict['time']
        self.injected_population = {col : val[0] for col,val in data.loaded_dict['population_parameters'].to_dict().items()} #self.loaded_dict['injected_population']
        self.limits = self.loaded_dict['limits']
        self.duration = self.loaded_dict['duration']
        
    def plot_psd(self):
        f, psd = calculate_psd(self.strain, self.time, self.sample_rate)
        fig, ax = plt.subplots(dpi=150)
        plt.plot(f, psd)
        plt.xscale("log")
        plt.yscale("log")
        plt.show()
        
    def plot_strain(self):
        fig, ax = plt.subplots(dpi=150)
        plt.plot(self.time, self.strain)
        plt.xlabel(r"$t (s)$")
        plt.ylabel(r"strain")
        plt.title(f"Strain")
        plt.grid(alpha=0.3)
        plt.show()
