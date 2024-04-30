import numpy as np
from astropy.constants import GM_sun as msun
from astropy.constants import G as Gnewton
from astropy.constants import c
from astropy.units import parsec, au


class foreground():


    def __init__(self, snr_thresh):
        self.snr_thresh = snr_thresh

        self.base_population = self.__base_population()

    def psd(self, alpha, beta, Ntot):

        
        
        return psd
    
    
    def _base_population(self):

        ## mc : 0.5 - 1.4:
        ## dl = 1 - 50 kpc

        mc_draw = np.random.uniform(low=0.5, high=1.4, size=int(5e4)) * msun
        r_draw = np.random.uniform(low=1, high=50, size=int(5e4)) * au
        d_draw = np.random.power(low=1, high=50, size=int(5e4)) * 1e3 * parsec

        f_draw = self.calc_freqs(mc_draw, r_draw)

        amplitude_draw = self.calc_amplitudes(mc_draw, f_draw, d_draw)
        import pdb; pdb.set_trace()

        base_population = {}

        base_population['mc'] = mc_draw
        base_population['r'] = r_draw
        base_population['d'] = d_draw
        base_population['amplitude'] = amplitude_draw

        return base_population

    def calc_freqs(self, mc, r):

        return np.sqrt(Gnewton * mc / (np.pi * r**3))

    def calc_ampltidues(self, mc_draw, f_draw, d_draw):

        return (4/d_draw) * (Gnewton*mc_draw)**(5/3) \
                (np.pi * f_draw / c)**(2/3)

        





