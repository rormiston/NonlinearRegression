from checkSource import checkSource
checkSource()

import math
import os
import threading, subprocess
import numpy as np

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})
from matplotlib import pyplot as plt

#import pymultinest
from pyswarm import pso
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


lb = [1.0,1.0,0.0,0.0]
ub = [11.0,11.0,1.0,5.0]


def myloglike(cube):
        Nepoch    = int(np.round(cube[0]))
        Nlayers   = int(np.round(cube[1]))
        dropout   = cube[2]
        cost_func = get_cost_func(int(np.round(cube[3])))

        kwargs = {"Nepoch": Nepoch,
                  "doCBC": False,
                  "Nlayers": Nlayers,
                  "dropout": dropout,
                  "verbose": 0,
                  "fs_slow": 32,
                  "doPlots": False,
                  "doParallel": True,
                  "gpu_count" : 4,
                  "cost_func" : cost_func,
                  "doLoadDownsampled": True}

        print Nepoch, Nlayers, dropout, cost_func
        prob = run_test(**kwargs)

        if not np.isfinite(prob):
            prob = 0.0

	return prob


# In[13]:
plotDirectory = 'plots_swarm'
if not os.path.isdir(plotDirectory):
    os.mkdir(plotDirectory)

xopt, fopt = pso(myloglike, lb, ub,
                 args    = (),
                 omega   = 0.5,
                 phip    = 0.5,
                 phig    = 0.5,
                 kwargs  = {},
                 debug   = False,
                 ieqcons = [],
                 maxiter = 100,
                 minstep = 1e-8,
                 minfunc = 1e-8,
                 f_ieqcons = None,
                 swarmsize = 100)

print xopt, fopt
