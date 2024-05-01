# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:21:28 2024

@author: Alexander
"""


import noise
import likelihood
import foreground
from datageneration import load_data ## or whatever 
from joint_likelihood import init_engine
import plotting_chains
import argparse
import os


def run_gb_inference(datafile,outdir,snr_thresh=10):
    ### load the data
    ## Asad will write a thing to do this
    data = load_data(datafile)
    
    ## get the data frequencies
    fbins = data.fs
    
    
    ## initialize the population model
    gb_pop_model = foreground.Foreground(fbins,snr_thresh=snr_thresh)
    
    ## intialize the sampler
    engine = init_engine(*args)
    
    ## run the sampler
    
    
    ## get the samples
    samples = engine.get_samples()
    
    ## generate plots
    plotting_chains(samples,outdir,**kwargs)
    
    
    
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
    



