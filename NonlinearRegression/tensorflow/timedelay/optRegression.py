from checkSource import checkSource
checkSouce()

import math
import os
import threading, subprocess
import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})
from matplotlib import pyplot as plt
import json

import pymultinest
from bilinearRegression import run_test

# our probability functions
# Taken from the eggbox problem.

# Make sure the data loads when running from any directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)


def get_cost_func(idx):
    cost_funcs = ['mean_squared_error',
                  'mean_absolute_error',
                  'mean_absolute_percentage_error',
                  'mean_squared_logarithmic_error',
                  'squared_hinge',
                  'hinge',
                  'poisson']

    return cost_funcs[idx]


def myprior(cube, ndim, nparams):
        cube[0] = cube[0]*10.0 + 1.0
        cube[1] = cube[1]*10.0 + 1.0
        cube[2] = cube[2]*1.0
        cube[3] = cube[3]*5.0


def myloglike(cube, ndim, nparams):
        Nepoch    = int(np.round(cube[0]))
        Nlayers   = int(np.round(cube[1]))
        dropout   = cube[2]
        cost_func = get_cost_func(int(np.round(cube[3])))

        kwargs = {"Nepoch" : Nepoch,
                  "doCBC"  : False,
                  "Nlayers": Nlayers,
                  "dropout": dropout,
                  "verbose": 0,
                  "fs_slow": 32,
                  "doPlots": False,
                  "gpu_count" :4,
                  "cost_func" :cost_func,
                  "doParallel": True,
                  "doLoadDownsampled": True}

        print Nepoch, Nlayers, dropout, cost_func
        prob = run_test(**kwargs)

        if not np.isfinite(prob):
            prob = 0.0

	return prob


# In[13]:
plotDirectory = 'plots_opt'
if not os.path.isdir(plotDirectory):
    os.mkdir(plotDirectory)

# number of dimensions our problem has
parameters = ["Nepoch","Nlayers","dropout","cost_funcs"]
n_params   = len(parameters)

n_live_points      = 10
evidence_tolerance = 1.0

# run MultiNest
pymultinest.run(myloglike,
                myprior,
                n_params,
                resume  = True,
                verbose = True,
                n_live_points        = n_live_points,
                sampling_efficiency  = 'model',
                evidence_tolerance   = evidence_tolerance,
                outputfiles_basename = '%s/2-'%plotDirectory,
                importance_nested_sampling = False)

# lets analyse the results
analyzer = pymultinest.Analyzer(n_params = n_params,
                                outputfiles_basename = '%s/2-'%plotDirectory)
stats = analyzer.get_stats()

# store name of parameters, always useful
with open('%sparams.json' % analyzer.outputfiles_basename, 'w') as f:
	json.dump(parameters, f, indent=2)

# store derived stats
with open('%sstats.json' % analyzer.outputfiles_basename, mode='w') as f:
	json.dump(stats, f, indent=2)

print('\n{0} ANALYSIS {0}'.format('-' * 30))
print("Global Evidence:\n\t%.15e +- %.15e" % ( stats['nested sampling global log-evidence'], stats['nested sampling global log-evidence error'] ))
