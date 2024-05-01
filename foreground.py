import jax.numpy as jnp
import numpy as np
from astropy.constants import GM_sun as gmsun
from astropy.constants import G as Gnewton
from astropy.constants import c as cspeed
import astropy.units as u
from astropy.units import parsec, au

xp = np

class foreground():


    def __init__(self, fbins, snr_thresh=10):
        self.snr_thresh = snr_thresh
        self.fbins = fbins
        self.delf = fbins[1] - fbins[0]
        self.fmins_ub = fbins + 0.5*self.delf
        self.fmins_lb = fbins - 0.5*self.delf

        self.base_population = self._base_population()

    def psd(self, alpha, beta, Ntot):

        
        
        return psd
    
    
    def _base_population(self):

        ## mc : 0.5 - 1.4:
        mc_draw = np.random.uniform(low=0.5, high=1.4, size=int(5e4)) * gmsun / Gnewton
        r_draw = np.random.uniform(low=1, high=50, size=int(5e4)) * au

        ## dl = 1 - 50 kpc
        d_draw = np.sqrt(1**2 + np.random.uniform(size=int(5e4)) * \
                         (50**2 - 1**2) ) * 1e3 * parsec

        r_draw, d_draw = r_draw.to(u.m), d_draw.to(u.m)

        f_draw = self.calc_freqs(mc_draw, r_draw)

        A_draw = self.calc_amplitudes(mc_draw, f_draw, d_draw)

        base_population = {}
    
        base_population['mc'] = mc_draw
        base_population['r'] = r_draw
        base_population['d'] = d_draw
        base_population['A'] = A_draw

        return base_population

    def calc_freqs(self, mc, r):

        return xp.sqrt(Gnewton * mc / (xp.pi * r**3))

    def calc_amplitudes(self, mc, f, d):

        amplitude =  (4/d) * (Gnewton*mc)**(5/3) * (xp.pi * f / cspeed)**(2/3)

        return amplitude



if __name__ == "__main__":
    fg = foreground()



