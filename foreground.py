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


        self.mc_min, self.mc_max = 0.5, 1.5 ## in solar masses
        self.r_min, self.r_max =  3.34e-4, 1.3e-2 ## in AU
        self.d_min, self.d_max = 1, 50 ## in kpc

        self.base_population = self._base_population()


    def psd(self, alpha, beta, Ntot):



        return psd

    def get_pop_wts(self, alpha, beta):

        log_p_mc = jnp.log(alpha + 1) + alpha * jnp.log(self.base_population['mc']) - \
        jnp.log( jnp.power(self.mc_max, alpha + 1) - jnp.power(self.mc_min, alpha + 1))

        log_p_r  = jnp.log(beta + 1) + beta * jnp.log(self.base_population['r']) - \
        jnp.log( jnp.power(self.r_max, beta + 1) - jnp.power(self.r_min, beta + 1))

        log_p_d = jnp.log(2) + jnp.log(self.base_population['d']) - \
        jnp.log(jnp.power(self.d_max, 2) - jnp.power(self.d_min, 2)) 

        return jnp.exp(log_p_mc + log_p_r + log_p_d - self.base_population['log_priors'])

    def _base_population(self):

        ndraw = int(5e4)
        mc_draw = np.random.uniform(low=self.mc_min, high=self.mc_max, size=ndraw) 
        r_draw = np.random.uniform(low=self.r_min, high=self.r_max, size=ndraw)
        d_draw = np.sqrt(self.d_min**2 + np.random.uniform(size=ndraw) * \
                         (self.d_max**2 - self.d_min**2) )

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
        base_population['f'] = f_draw.value

       
        base_population['log_priors'] =  - np.log(self.mc_max - self.mc_min) - \
                                        np.log(self.r_max - self.r_min) - \
                                        np.log(2) + np.log(d_draw) - \
                                        np.log(self.d_max**2 - self.d_min**2) 

        # Masking by freq bin
        fmask = np.zeros((len(fbins),ndraw))
        import pdb
        pdb.set_trace()
        for i in enumerate(fbins):
            fub = self.fmins_ub[i[0]]
            flb = self.fmins_lb[i[0]]
            mask = (f_draw >= flb)*(f_draw < fub)
            fmask[i[0]] = mask
            print(base_population['A'][mask])
        return fmask, base_population

    def calc_freqs(self, mc, radius):

        mc = mc * gmsun / Gnewton
        radius = radius * au

        return xp.sqrt(Gnewton * mc / (xp.pi * radius.to(u.m)**3))

    def calc_amplitudes(self, mc, f, dl):

        mc = mc * gmsun / Gnewton
        dl = dl*1e3*parsec

        amplitude =  (4/dl.to(u.m)) * (Gnewton*mc)**(5/3) * (xp.pi * f / cspeed)**(2/3)

        return amplitude



if __name__ == "__main__":
    fbins = (10**(np.linspace(-11,-6,10)))/u.s
    fg = foreground(fbins)



