
# for now this is very schematic but should work
# could jax this
def loglikelihood(data,params):
    # assume data is freq domain
    residual = data - generate_resolvable_waveforms(params)
    resp_A, resp_E, resp_T = get_responses() # these are sky-averaged for now?
    unresoved_theoretical_psd = get_pop_psd(params)
    cov_A = noise_psdA + resp_A*unresoved_theoretical_psd
    cov_E = noise_psdE + resp_E*unresoved_theoretical_psd
    cov_T = noise_psdT + resp_T*unresoved_theoretical_psd
    logdetC = np.log(cov_A*cov_E*cov_T)

    return np.sum( -0.5*np.log(2*np.pi) - 0.5*logdetC - np.abs(residual_A)**2 / cov_A - np.abs(residual_E)**2/cov_E - np.abs(residual_T)**2/cov_T)
 
