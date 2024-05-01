#!/bin/env python3
"""
Created on Wed May  1 12:21:28 2024

@author: Alexander W. Criswell
"""

## imports
import foreground
from datageneration import load_data ## or whatever 
from joint_likelihood import init_engine
from plotting_chains import plot_chains
import argparse
import os

## NEED TO ALSO IMPORT THE LIKELIHOOD CODE


def run_gb_inference(datafile,outdir,snr_thresh=10):
    ### load the data
    ## Asad will write a thing to do this
    data = load_data(datafile)
    
    ## get the data frequencies
    fbins = data.fs
    
    
    ## initialize the population model
    gb_pop_model = foreground.Foreground(fbins,snr_thresh=snr_thresh)
    
    ## intialize the sampler
    args = [] ## fill in sampling args
    engine = init_engine(*args)
    
    ## run the sampler
    end_state = engine.run_mcmc()
    
    ## get the samples
    samples = engine.get_samples()
    
    ## generate plots
    kwargs = {truths = [], #TODO: truth values of resolved GBs + population params
              plotdir = outdir} ## fill in plotting dict
    plot_chains(samples,**kwargs)
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run LISA GB population inference.')
    parser.add_argument('datafile', type=str, help='/path/to/data/file.hdf5')
    parser.add_argument('--snr_thresh', type=float, help='SNR threshold between resolved and unresolved signals.', default=10)
    parser.add_argument('--outdir', type=str, help='/path/to/save/dir/', default=None)
    
    if parser.outdir is None:
        outdir = '.'
    else:
        os.mkdir(parser.outdir)
        outdir = parser.outdir
    
    run_gb_inference(parser.datafile,outdir,snr_thresh=parser.snr_thresh)
    



