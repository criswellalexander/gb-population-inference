#!/bin/env python3
"""
Created on Wed May  1 12:21:28 2024

@author: Alexander W. Criswell
"""

## imports
import foreground
from datageneration import DataLoader
from utils import generate_time_domain_detector_noise_with_seed, get_rfft, generate_detector_noise_psd_with_seed
from joint_likelihood import init_engine
from plotting_chains import plot_chains
import argparse
import os

## NEED TO ALSO IMPORT THE LIKELIHOOD CODE


def run_gb_inference(datafile,outdir,snr_thresh=10, noise_amplitude=1e-21, noise_seed=0, nresolved=5, duration=10_000):
    ### load the data
    data = DataLoader(datafile)

    ### Get the true injected values
    truths = data.population_parameters

    ## get the data frequencies
    waveform, times, sample_rate = data.strain, data.time, data.sample_rate
    noise = generate_time_domain_detector_noise_with_seed(times, noise_amplitude, seed=noise_seed)
    strain = waveform + noise

    fbins, fft_data = get_rfft(strain, times, sample_rate)
    noisePSD = generate_detector_noise_psd_with_seed(sample_rate, duration, noise_amplitude, seed=noise_seed)
    
    
    ## initialize the population model
    gb_pop_model = foreground.foreground(noisePSD, fbins, nresolved=nresolved ,snr_thresh=snr_thresh)
    
    ## intialize the sampler
    args = [] ## fill in sampling args
    engine = init_engine(*args)
    
    ## run the sampler
    end_state = engine.run_mcmc()
    
    ## get the samples
    samples = engine.get_samples()
    
    ## generate plots
    kwargs = dict(truths = [], #TODO: truth values of resolved GBs + population params
                  plotdir = outdir) ## fill in plotting dict
    plot_chains(samples,**kwargs)
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run LISA GB population inference.')
    parser.add_argument('datafile', type=str, help='/path/to/data/file.hdf5')
    parser.add_argument('--snr_thresh', type=float, help='SNR threshold between resolved and unresolved signals.', default=10)
    parser.add_argument('--outdir', type=str, help='/path/to/save/dir/', default=None)
    parser.add_argument('--noise_seed', type=int, help='Seed for data noise generation', default=None)
    parser.add_argument('--noise_amplitude', type=float, help='data noise amplitude', default=None)
    parser.add_argument('--nresolved', type=int, help='number of resolved signals', default=None)
    parser.add_argument('--duration', type=int, help='data duration in seconds', default=None)

    args = parser.parse_args()
    
    if args.outdir is None:
        outdir = '.'
    else:
        if not os.path.exists(args.outdir):
            os.mkdir(args.outdir)
        outdir = args.outdir

    noise_seed = args.noise_seed or 0
    noise_amplitude = args.noise_seed or 1e-21
    nresolved = args.nresolved or 5
    duration = args.duration or 10_000
    
    run_gb_inference(args.datafile,outdir,snr_thresh=args.snr_thresh, 
                                          noise_seed=noise_seed, 
                                          noise_amplitude=noise_amplitude,
                                          nresolved=nresolved,
                                          duration=duration)
    



