from chainconsumer import ChainConsumer, Chain, PlotConfig, Truth
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import os
import pandas as pd

def plot_chains(Nresolved : int, chains, pop_param_labels : list[str] = [r'$\alpha$', r'$\beta$'], truths = [], plotfolder : str = 
pathlib.Path(__file__).parent.absolute() ):
    labels = []
    for i in range(Nresolved):
        labels.extend([f'$A_{{{i}}}$', f'$f_{{{i}}}$', f'$\\phi_{{{i}}}$'])
    labels.extend(pop_param_labels)
    df = pd.DataFrame(chains, columns = labels)
    c = ChainConsumer()
    c.add_chain(Chain(name='samples',samples=df))
    c.add_truth(Truth(location={l: t for l,t in zip(labels,truths)}))
    fig = c.plotter.plot(filename=os.path.join(plotfolder,'chains.pdf'))



from scipy.stats import multivariate_normal
Nresolved = 3

means = [1e-20,1e-3,0.5]*Nresolved + [1.0,1.0]
cov = np.diag(1e-4*np.ones_like(means))
fake_chains = multivariate_normal(mean=means,cov=cov,allow_singular = False).rvs(10000)
plot_chains(Nresolved, fake_chains, truths = means)
