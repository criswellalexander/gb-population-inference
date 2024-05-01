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
        ndraw = int(5e4)
        mc_draw = np.random.uniform(low=0.5, high=1.4, size=ndraw) * gmsun / Gnewton
        r_draw = np.random.uniform(low=1, high=50, size=ndraw) * au

        ## dl = 1 - 50 kpc
        d_draw = np.sqrt(1**2 + np.random.uniform(size=int(5e4)) * \
                         (50**2 - 1**2) ) * 1e3 * parsec

        r_draw, d_draw = r_draw.to(u.m), d_draw.to(u.m)

        f_draw = self.calc_freqs(mc_draw, r_draw)


        A_draw = self.calc_amplitudes(mc_draw, f_draw, d_draw)


        base_population = {}

        # Sorting everything by amplitude ahead of masking by freq bin
        sort_mask = np.argsort(A_draw)
        A_draw = A_draw[sort_mask]
        f_draw = f_draw[sort_mask]
        mc_draw = mc_draw[sort_mask]
        r_draw = r_draw[sort_mask]
        d_draw = d_draw[sort_mask]

        base_population['mc'] = mc_draw
        base_population['r'] = r_draw
        base_population['d'] = d_draw
        base_population['A'] = A_draw

        # Masking by freq bin
        fmask = np.zeros((len(fbins),ndraw))
        for i in enumerate(fbins):
            fub = self.fmins_ub[i[0]]
            flb = self.fmins_lb[i[0]]
            mask = (f_draw >= flb)*(f_draw < fub)
            fmask[i[0]] = mask
            #print(base_population['A'][mask])
        return fmask, base_population

    def calc_freqs(self, mc, r):

        return xp.sqrt(Gnewton * mc / (xp.pi * r**3))

    def calc_amplitudes(self, mc, f, d):

        amplitude =  (4/d) * (Gnewton*mc)**(5/3) * (xp.pi * f / cspeed)**(2/3)

        return amplitude



if __name__ == "__main__":
    fbins = (10**(np.linspace(-11,-6,10)))/u.s
    fg = foreground(fbins)



