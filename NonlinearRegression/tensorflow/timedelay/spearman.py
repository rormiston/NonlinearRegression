"""
This script calculates "Spearman's rho" and the Pearson coefficient
for every combination of products of witness channels with DARM. Any channel
with a correlation above the given threshold for the given parameter (pearson/
spearman coefficient) is written to a csv file. NOTE: the background channel is
ignored for both real and mock data.
"""
from __future__ import division
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import seaborn as sns
import NonlinearRegression.tools.preprocessing as ppr
from itertools import chain, combinations
from scipy.stats import spearmanr
from scipy import stats


# Scale the arrays to (-1, 1)
def feature_scaling(array):
    mx = np.max(np.abs(array))
    mn = np.min(np.abs(array))
    array = (array - mn) / mx
    return array


# Get every combination of witness channels
def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))


def spearman(data_type = 'real',
             ifo       = 'L1',
             outputDir = './',
             output    = 'spearman_results.csv',
             rho       = 0.3,
             pear      = 0.99,
             threshold = 'pearson'):

    # Get the data to test
    datafile   = ppr.get_datafile(None, data_type,  ifo=ifo)
    dataset, _ = ppr.get_dataset(datafile, data_type=data_type)
    darm       = dataset[:, 0]
    witnesses  = dataset[:, 2:]
    Nwit       = witnesses.shape[1]
    chans      = list(range(2, Nwit))

    print('Dataset: {0}'.format(datafile))

    # Scale darm
    darm = feature_scaling(darm)

    # Collect all of the witness channel combinations
    all_combos = []
    for subset in all_subsets(chans):
        all_combos.append(subset)

    # Don't include an empty list (only one of those)
    all_combos = [x for x in all_combos if len(x) > 0]

    # Test all combinations
    spearman_corr = {}
    for index, combo in enumerate(all_combos):

        # Monitor progress
        index += 1
        sys.stdout.write('\rRunning combo {0}/{1}'.format(index, len(all_combos)))
        sys.stdout.flush()
        if index == len(all_combos):
            print('')

        # Take every combination of products of witness channels
        temp = np.ones_like(darm)
        for wit in combo:
            temp *= witnesses[:, wit]

        #Scale and do correlation tests
        temp = feature_scaling(temp)
        correlation, pearson = spearmanr(darm, temp)
        spearman_corr[combo] = (correlation, pearson)

    # Write the results to a file
    count = 0
    max_corr = 0
    max_pear = 0
    FILE = '{0}/{1}'.format(outputDir, output)

    # Make sure the path and file exist
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    if not os.path.isfile(FILE):
        os.system('touch {}'.format(FILE))

    with open(FILE, 'w') as f:
        f.write('Datafile: ' + datafile + '\n')
        f.write('Channel\tCorrelation\tPearson\n')
        for chan_combo, sp in spearman_corr.items():
            if threshold == 'pearson':
                cutoff = np.abs(sp[0]) >= rho
            elif threshold == 'rho':
                cutoff = np.abs(sp[1]) >= pear
            elif threshold == 'and':
                cutoff = (np.abs(sp[0]) >= rho) and (np.abs(sp[1]) >= pear)
            elif threshold == 'or':
                cutoff = (np.abs(sp[0]) >= rho) or (np.abs(sp[1]) >= pear)
            else:
                print('ERROR: Threshold keyword not understood')
                print('Allowable keywords: "pearson", "rho", "and", "or"')
                sys.exit(1)

            if cutoff:
                count += 1
                f.write('{0}\t{1}\t{2}\n'.format(chan_combo, sp[0], sp[1]))

            # Find max
            if np.abs(sp[0]) > np.abs(max_corr):
                max_corr = sp[0]
            if np.abs(sp[1]) > np.abs(max_pear):
                max_pear = sp[1]

    # Print a summary
    combination = 'combinations'

    if count == 0:
        block = '[-]'
        written_to_file = '[-] Nothing written to {0}'.format(output)
    elif count > 0:
        block = '[+]'
        written_to_file = '[+] Results written to {0}'.format(output)
        if count == 1:
            combination = 'combination'

    message = ('{0} {1} channel {2} had an absolute value spearman coefficient '
               'over {3}. \n'.format(block, count, combination, rho))

    max_str = '[+] Max Spearman/Pearson coefficients {0}/{1}.\n'.format(max_corr,
                                                                        max_pear)
    summary = max_str + message + written_to_file
    print(summary)
    return FILE


def spearplot(textfile):

    # Read the channel combinations
    combinations = []
    nums = re.compile(r'\d+')
    with open(textfile) as f:
        lines = f.readlines()
        datafile = lines[0].strip('\n').split(' ')[1].strip()
        for line in lines[2:]:
            channels = line.split('\t')[0]
            channels = nums.findall(channels)
            if len(channels) > 0:
                combinations.append(channels)

    # Get the dataset used
    dataset, _ = ppr.get_dataset(datafile)
    darm = feature_scaling(dataset[:, 0])

    # Pick out the channel data from the names
    for ix, combination in enumerate(combinations):
        print('Making plot {0}/{1}'.format(ix + 1, len(combinations)))
        temp = np.ones(shape=(dataset.shape[0]))
        for chan in combination:
            temp *= dataset[:, int(chan)]
        temp = feature_scaling(temp)

        # Plot them
        plot_data = {'DARM':darm, 'Channels':temp}
        df = pd.DataFrame(plot_data)
        G = sns.PairGrid(df)

        G.map_diag(sns.distplot,
                   fit   = stats.gamma,
                   kde   = False,
                   rug   = False,
                   color = 'k')

        cmap = sns.cubehelix_palette(as_cmap = True,
                                     dark    = 0,
                                     light   = 1,
                                     reverse = True)
        G.map_offdiag(sns.kdeplot,
                      cmap         = cmap,
                      shade        = True,
                      shade_lowest = False)

        plt.suptitle('Spearman Results')
        plt.savefig('Spearman_channel_corr_{0}.png'.format(ix))
        plt.close()


if __name__ == '__main__':


    def parse_command_line():
        """
        parse command line
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("--data_type", "-d",
                            help    = "real or mock data set",
                            default = "real",
                            dest    = "data_type",
                            type    = str)

        parser.add_argument("--ifo", "-ifo",
            				help    = "L1 or H1",
                            default = "L1",
            				dest    = "ifo",
                            type    = str)

        parser.add_argument("--outputDir", "-dir",
            				help    = "directory in which to store results",
                            default = "./",
            				dest    = "outputDir",
                            type    = str)

        parser.add_argument("--output", "-o",
            				help    = "output file name",
                            default = "spearman_results.csv",
            				dest    = "output",
                            type    = str)

        parser.add_argument("--rho", "-r",
            				help    = "spearman's rho threshold value",
                            default = 0.30,
            				dest    = "rho",
                            type    = float)

        parser.add_argument("--pearson", "-p",
            				help    = "pearson's coefficient threshold value",
                            default = 0.99,
            				dest    = "pear",
                            type    = float)

        parser.add_argument("--threshold", "-t",
                            help    = "chose from: pearson, rho, and, or",
                            default = "rho",
            				dest    = "threshold",
                            type    = str)

        params = parser.parse_args()

        # Convert params to a dict
        model_params = {}
        for arg in vars(params):
            model_params[arg] = getattr(params, arg)

        return model_params


    # Get the flags
    model_params = parse_command_line()

    # Run the correlation test
    FILE = spearman(**model_params)

    # Plot the results
    spearplot(FILE)
