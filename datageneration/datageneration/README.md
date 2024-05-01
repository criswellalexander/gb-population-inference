# datageneration

A micro package to generate a sampling of time series waveforms for Populations of White Dwarf binaries. For now we are generating test waveforms with each white dwarf system contributing a sinosoidal contribution to the time series. 

## Implemented Models:

#### Power Law in chirp mass and Power Law in seperation

$$p(\mathcal{M}_c, r | \alpha, \beta) \propto \mathcal{M}_c^{\alpha} r^\beta$$

## How to Use:

First lets import the white dwarf population model we want to use

```python
from datageneration import PowerLawChirpPowerLawSeperation
```

then, define the limits of the WD parameters

```python
import numpy as np

## Define the ranges over which the population distribution exists

limits = {'chirp_mass':          [0.5, 1.4],  # In solar masses
          'seperation':          [0.5, 20],   # In 1e8 meters
          'luminosity_distance': [1, 50],     # In kilo parsecs
          'phase':               [0, 2*np.pi]}

## Instantiate the population distribution object

dist = PowerLawChirpPowerLawSeperation(limits=limits, 
                                       distance_power_law_index=1)   # p(d) ~ d
```

Now lets generate some samples from the above distribution

```python
# Get samples from a given hyperparameter set
import pandas as pd

Lambda = {'alpha' : 3.0, 'beta' : -2.0}
pd.DataFrame(dist.generate_samples(Lambda, size=10))
```
|    |   chirp_mass |   seperation |   luminosity_distance |    phase |\n|---:|-------------:|-------------:|----------------------:|---------:|\n|  0 |     1.34114  |     3.44976  |               49.8525 | 2.41405  |\n|  1 |     1.23741  |     0.885718 |               30.2443 | 5.18609  |\n|  2 |     1.24095  |     3.24956  |               46.3433 | 5.71657  |\n|  3 |     1.34529  |     1.7046   |               43.7225 | 1.20969  |\n|  4 |     1.34794  |     0.549323 |               20.2949 | 1.29271  |\n|  5 |     1.30497  |     2.94658  |               13.5384 | 0.198767 |\n|  6 |     1.21342  |     1.2791   |               30.1709 | 1.30623  |\n|  7 |     0.685329 |     0.540291 |               38.4316 | 4.61342  |\n|  8 |     1.17888  |     0.605956 |               35.5994 | 5.19118  |\n|  9 |     1.16896  |     1.72301  |               25.4811 | 5.28927  |

We can also generate a waveform with exactly `1_000` White Dwarfs drawn from the above distribution.

```python
Lambda = {'alpha' : 3.0, 'beta' : -2.0}
ts, strain = dist.generate_time_series(Lambda, N_white_dwarfs=1000)
dist.plot_time_series(ts, strain)
```

![Alt text](./imgs/waveform_sample.png)
