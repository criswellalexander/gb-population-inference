from lisatools.detector import EqualArmlengthOrbits, ESAOrbits, LISAModel
from lisatools.sensitivity import get_sensitivity, AET1SensitivityMatrix
import lisatools.detector as lisa_models
from matplotlib import pyplot as plt
import numpy as np
import scipy

def randc(r):
    return r/np.sqrt(2)*(np.random.randn(*r.shape) + 1j*np.random.randn(*r.shape))

def generate_noise_realization(dt,Tobs,noisemodel : LISAModel = lisa_models.sangria, seed = 0, DEBUG = False):
    Nt = Tobs // dt
    Nchannels = 3
    freqs = np.fft.rfftfreq(Nt,dt)
    N = len(freqs)
    sens_mat = AET1SensitivityMatrix(
        freqs,
        model=noisemodel,
    )
    SnA,SnE,SnT = sens_mat.sens_mat
    retdict = {"expected_PSD": np.zeros((Nchannels,N)), "fft": np.zeros((Nchannels,N),dtype=np.complex128),  "freqs":freqs, 'timeseries' : np.zeros((Nchannels,Nt))}
    retdict['expected_PSD'][0,:] = SnA
    retdict['expected_PSD'][1,:] = SnE
    retdict['expected_PSD'][2,:] = SnT
    retdict['fft'][:,:] = randc(np.sqrt(retdict['expected_PSD']))
    # method based on https://dsp.stackexchange.com/questions/83744/how-to-generate-time-series-from-a-given-one-sided-psd
    retdict['fft'][:,0] = 0.0
    fake_fft = retdict['fft']
    P_fft_one_sided = np.abs(fake_fft)**2
    N_P = P_fft_one_sided.shape[1]
    Nf = 2*(N_P-1)
    P_fft_new = np.zeros((Nchannels,Nf), dtype=complex)
    P_fft_new[:,0:int(Nf/2)+1] = P_fft_one_sided
    P_fft_new[:,int(Nf/2)+1:] = P_fft_one_sided[:,-2:0:-1]
    phases = np.random.uniform(0, 2*np.pi, (Nchannels,int(Nf/2)))
    X_new = np.sqrt(P_fft_new)
    X_new[:,1:int(Nf/2)+1] = X_new[:,1:int(Nf/2)+1] * np.exp(2j*phases)
    X_new[:,int(Nf/2):] = X_new[:,int(Nf/2):] * np.exp(-2j*phases[:,::-1])
    X_new = X_new * np.sqrt(N_P/2/np.sqrt(2*np.pi))

    x_new = np.real(scipy.fft.ifft(X_new))
    if DEBUG:
        plt.loglog(freqs,np.abs(fake_fft[0])**2,label='dc to nyquist actual PSD A')
        plt.loglog(freqs,np.abs(fake_fft[2])**2,label='dc to nyquist actual PSD T')
        f, psdA = scipy.signal.welch(x_new[0,:],fs=1/dt,nperseg=len(x_new[0]),window='boxcar',noverlap=0)
        f, psdT = scipy.signal.welch(x_new[2,:],fs=1/dt,nperseg=len(x_new[0]),window='boxcar',noverlap=0)

        plt.loglog(f,psdA,label='demo timeseries PSD A',ls='dotted')
        plt.loglog(f,psdT,label='demo timeseries PSD T',ls='dotted')
        plt.loglog(freqs,SnA,label='dc to nyquist theoretical PSD A')
        plt.loglog(freqs,SnT,label='dc to nyquist theoretical PSD T')
        plt.legend()
        plt.show()

        plt.plot(np.arange(Nf)*dt,x_new[0], label='timeseries (A)')
        plt.xlabel("t [s]")
        plt.legend()
        plt.show()
    # WARNING: timeseries is not band-limited, goes from dc to nyquist. Should match ffts' PSD with a boxcar window

    retdict["timeseries"] += x_new
    retdict["ts"] = np.linspace(0,Nt*dt,Nt)
    retdict['PSD'] = np.abs(retdict['fft'])**2
    return retdict
if __name__ == '__main__':
    generate_noise_realization(5,30*24*3600,DEBUG=True)
