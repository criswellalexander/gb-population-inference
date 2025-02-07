import jax.numpy as jnp
import numpy as np
from astropy.constants import GM_sun as gmsun
from astropy.constants import G as Gnewton
from astropy.constants import c as cspeed
import astropy.units as u
from astropy.units import parsec, au

xp = np

class foreground():


    def __init__(self, noisePSD, fbins, nresovled=5, snr_thresh=10):
        self.snr_thresh = snr_thresh
        self.fbins = fbins
        self.noisePSD = noisePSD
        self.delf = fbins[1] - fbins[0]
        self.fmins_ub = fbins + 0.5*self.delf
        self.fmins_lb = fbins - 0.5*self.delf
        self.nresolved = nresovled

        self.mc_min, self.mc_max = 0.5, 1.5 ## in solar masses
        self.r_min, self.r_max =  3.34e-4, 1.3e-2 ## in AU
        self.d_min, self.d_max = 1, 50 ## in kpc
        self.ndraw = int(1e5)


        self.fmask, self.base_population = self._base_population()


    def psd(self, alpha, beta, Ntot):

        wts = self.get_pop_wts(alpha, beta)

        nunresolved = Ntot - self.nresolved

        Nij = [self.calc_Nij(self.base_population['A'][i], self.noisePSD[i], wt) for i, wt in enumerate(wts)]

        import pdb; pdb.set_trace()

        prop_factor = Ntot * jnp.sum()

        prop_factor = Ntot * self.fmask.sum(axis=1) 



        return psd

    def get_pop_wts(self, alpha, beta):

        log_p_mc_constant = jnp.log(jnp.abs(alpha + 1)) - \
        jnp.log(jnp.abs(jnp.power(self.mc_max, alpha + 1) - jnp.power(self.mc_min, alpha + 1)))

        log_p_r_constant = jnp.log(jnp.abs(beta + 1)) -  \
        jnp.log(jnp.abs(jnp.power(self.r_max, beta + 1) - jnp.power(self.r_min, beta + 1)))

        log_p_d_constant = jnp.log(2) - jnp.log(jnp.power(self.d_max, 2) - jnp.power(self.d_min, 2))


        log_p_mc = [log_p_mc_constant + alpha * jnp.log(mc) for mc in self.base_population['mc']]
        log_p_r = [log_p_r_constant + beta * jnp.log(r) for r in self.base_population['r']]
        log_p_d = [log_p_d_constant + jnp.log(d) for d in self.base_population['d']]

        log_wts = [log_p_mc[i] + log_p_r[i] + log_p_d[i] - self.base_population['log_prior'][i] for i, mc in enumerate(log_p_mc)]

        wts = [jnp.exp(log_wt) for log_wt in log_wts]
        # Reintroducing this here! Needed since we are dealing with a matrix again.
        wts = np.where(self.base_population['mc'] == 0, 0, wts)
        wts = np.where(self.base_population['r'] == 0, 0, wts)
        wts = np.where(self.base_population['d'] == 0, 0, wts)
        #
        wt_sum = [jnp.sum(wt) for wt in wts]
        wt_sum = np.sum(wt_sum)   

        wts = [wt/wt_sum for wt in wts]     


        """
        log_p_mc = jnp.log(jnp.abs(alpha + 1)) + alpha * jnp.log(self.base_population['mc']) - \
        jnp.log(jnp.abs(jnp.power(self.mc_max, alpha + 1) - jnp.power(self.mc_min, alpha + 1)))

        log_p_r  = jnp.log(jnp.abs(beta + 1)) + beta * jnp.log(self.base_population['r']) - \
        jnp.log(jnp.abs(jnp.power(self.r_max, beta + 1) - jnp.power(self.r_min, beta + 1)))

        log_p_d = jnp.log(2) + jnp.log(self.base_population['d']) - \
        jnp.log(jnp.power(self.d_max, 2) - jnp.power(self.d_min, 2))

        wts =  jnp.exp(log_p_mc + log_p_r + log_p_d - self.base_population['log_priors'])
        
        wts = np.where(self.base_population['mc'] == 0, 0, wts)
        wts = np.where(self.base_population['r'] == 0, 0, wts)
        wts = np.where(self.base_population['d'] == 0, 0, wts)


        wts = wts / jnp.sum(wts)
        """

        return wts



    def _base_population(self):

        draw = {}
        draw['mc'] = np.random.uniform(low=self.mc_min, high=self.mc_max, size=self.ndraw)
        draw['r'] = np.random.uniform(low=self.r_min, high=self.r_max, size=self.ndraw)
        draw['d'] = np.sqrt(self.d_min**2 + np.random.uniform(size=self.ndraw) * \
                         (self.d_max**2 - self.d_min**2) )
        draw['f'] = self.calc_freqs(mc_draw, r_draw).value
        draw['A'] = self.calc_amplitudes(mc_draw, f_draw, d_draw).value

        # Sorting everything by amplitude ahead of masking by freq bin
        sort_mask = np.argsort(draw['A'])
        for param in draw.keys():
            draw[param] = draw[param][sort_mask]

        fmask = []
        # Masking by freq bin
        for i, fbin in enumerate(fbins):
            fub = self.fmins_ub[i]
            flb = self.fmins_lb[i]
            mask = (f_draw >= flb)*(f_draw < fub)
            fmask.append(mask)

        fmask = np.array(fmask)
        base_population = {}
        # Note! This creates an array where each column corresponds to the freq bin and each row to an individual binary.
        # The entry is zero if the binary is not in the freq bin.
        for param in draw.keys():
            base_population[param] = draw[param][:,None].T*fmask


        log_prior_constant =  - np.log(self.mc_max - self.mc_min) - \
                                np.log(self.r_max - self.r_min) - \
                                np.log(2)  - np.log(self.d_max**2 - self.d_min**2)
        base_population['log_prior'] = log_prior_constant + jnp.log(base_population['d'])

        #base_population['log_prior'] = [log_prior_constant + jnp.log(dl) for dl in d_sorted]

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


    @staticmethod
    def calc_Nij(A, noisePSD, wts):
        return A / (noisePSD + jnp.cumsum(wts*A))

if __name__ == "__main__":
    #fbins = (10**(np.linspace(-5,-2,10)))/u.s
    fbins = np.linspace(1e-4, 1e-2, 50)
    noisepsd = 1e-20 * np.ones(fbins.shape)    
    fg = foreground(noisepsd, fbins)
    fg.psd(-2, -1.5, 500)




