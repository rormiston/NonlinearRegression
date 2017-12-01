#!/usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
import inspect
import numpy as np
import pandas as pd

import os
# Hush tensorflow AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '5'

import keras.backend as K
import NonlinearRegression.tools.preprocessing as ppr
import re
import scipy.io as sio
import scipy.signal as sig
import seaborn as sns
import sys
import textwrap
from bs4 import BeautifulSoup
from cycler import cycler
from ConfigParser import ConfigParser
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


def write_summary(modelSummary,
                  version  = '0',
                  basename = 'LSTM',
                  PATH     = "./"):
    '''
    write_summary writes the standard out of model.summary()
    to a txt file

    Parameters
    ----------
        modelSummary: `keras.model.Model.summary`
            model summary object

        basename: `string`
            Name of the file running the command or the name of the model

        version: `int` or `str`
            variation of the model being run

        PATH: `string`
            output file path
    '''

    if '.' in basename:
        basename = basename.split('.')[0]

    PATH = PATH + "/params/{}/".format(basename)
    FILE = PATH + "{0}-{1}.txt".format(basename, version)

    # make sure output directory exists
    if not os.path.isdir(PATH):
        os.system('mkdir -p {0}'.format(PATH))

    # make sure file exists
    if not os.path.isfile(FILE):
        os.system('touch {}'.format(FILE))

    with open(FILE, 'w') as modSum:
        modelSummary(print_fn=lambda x: modSum.write(x + '\n'))


def get_opt_name(opt):
    '''
    get_opt_name takes in a keras optimizer object and
    returns its name

    Parameters
    ----------
        opt: `keras.optimizer`
            neural net's optimizer object

    Returns
    -------
        opt_name: `string`
            name of keras optimizer
    '''
    full_name = opt.__doc__.split('\n')[0]
    opt_name  = ' '.join(full_name.split(' ')[:-1])
    return opt_name


def write_run_params(model_params,
                     basename = 'LSTM',
                     version  = '0',
                     PATH     = './'):

    PATH = PATH + "/params/{}/".format(basename)
    if not os.path.isdir(PATH):
        os.system('mkdir -p {}'.format(PATH))

    FILE = PATH + "{0}-{1}.ini".format(basename, version)
    if not os.path.isfile(FILE):
        os.system('touch {}'.format(FILE))

        with open(FILE, 'w+') as f:
            f.write('[{0}]\n[{1}]'.format('run_test', 'optimizer'))

    # read and write to config file
    configs = ConfigParser()
    configs.read(FILE)

    opt_params = ['beta_1', 'beta_2', 'decay', 'epsilon', 'lr',
                  'momentum', 'nesterov', 'rho']

    for key, val in model_params.items():
        if key not in opt_params:
            configs.set('run_test', key, val)

    with open(FILE, 'w+') as configFile:
        configs.write(configFile)


def write_minimum_loss(minLoss  = 0,
                       name     = None,
                       PATH     = '.',
                       basename = 'LSTM',
                       version  = '0'):
    '''
    write_minimim_loss records the variance of the background
    for the current model and data and writes it to a file. This
    value is reused when making the webpages.

    Parameters
    ----------
        minLoss: `float`
            variance of the background

        fname: `string`
            name of the current file

        PATH: `string`
            path to model
    '''
    if '.' in name:
        name = name.split('.')[0]

    FILE = '{0}/params/{1}/minLoss-{2}.txt'.format(PATH, basename, version)

    if not os.path.isfile(FILE):
        os.system('touch {}'.format(FILE))

    with open(FILE, 'w') as f:
        f.write(str(minLoss))


def read_minimum_loss(minLossFile):
    '''
    read_minimum_loss reads and returns the minimum
    loss for a given file.

    Parameters
    ----------
        minLossFile: `string`
            full path to minimum loss file

    Returns
    -------
        minLoss: `string`
            minimum loss recorded from most recent run
            of the current model
    '''
    with open(minLossFile) as f:
        minLoss = f.readline().strip('\n')
    return minLoss


def write_channel_list(datafile,
                       PATH     = '.',
                       basename = 'LSTM',
                       version  = '0'):

    mat_file = sio.loadmat(datafile)

    if '.' in basename:
        basename = basename.split('.')[0]

    output = PATH + '/params/{0}/channel_list-{1}.txt'.format(basename, version)

    try:
        chans  = mat_file['chans']
        chans  = [str(c.strip()) for c in chans]

        with open(output, 'w') as f:
            for chan in chans:
                if chan == chans[-1]:
                    f.write(chan)
                else:
                    f.write(chan)
                    f.write('\n')
    except:
        with open(output, 'w') as f:
            if "bilinear" in datafile:
                f.write('Mock Data (bilinear)')

            elif "scatter" in datafile:
                f.write('Mock Data (scatter)')


def read_channel_list(chanlist):
    chans = []

    with open(chanlist) as f:
        lines = f.readlines()
        for line in lines:
            chan = line.strip('\n')
            chans.append(chan)

    return chans


def adjust_chan_list_length(model_dict):
    models = model_dict.keys()
    maxlen = 0

    # Find the longest channel list
    for model in models:
        chanlist = model_dict[model]['chan_list']
        chanlen  = len(chanlist)
        if chanlen >= maxlen:
            maxlen = chanlen

    # Add '-' to the short lists
    for model in models:
        chanlist = model_dict[model]['chan_list']

        if len(chanlist) < maxlen:
            for i in range(maxlen - len(chanlist)):
                chanlist.append('-')

            model_dict[model]['chan_list'] = chanlist

    return model_dict


def organize_run_data(summary      = None,
                      optimizer    = None,
                      opt_name     = None,
                      function     = None,
                      model_params = None,
                      PATH         = './',
                      name         = '',
                      minLoss      = 0,
                      move         = False,
                      datafile     = None):
    '''
    organize run_data combines some of the other functions
    in this module so that the regression scripts are minimally
    changed.

    Parameters
    ----------
        summary: `keras.model.Model.summary`
            keras model object

        optimizer: `keras.optimizers`
            keras optimizer object

        test: `function`
            run_test function
    '''

    # Get model basename
    regex_model = re.compile(r'[a-zA-Z]+')
    basename    = regex_model.findall(name)[0]
    regex_num   = re.compile(r'\d+')
    try:
        version = regex_num.findall(name)[0]
    except IndexError:
        version = '0'

    # Write model summary
    write_summary(summary,
                  version  = version,
                  basename = basename,
                  PATH     = PATH)

    # Write optimizer and run_test params
    write_run_params(model_params,
                     version  = version,
                     basename = basename,
                     PATH     = PATH)

    write_optimizer_params(opt_name, optimizer,
                           version  = version,
                           basename = basename,
                           PATH     = PATH)

    write_minimum_loss(minLoss  = minLoss,
                       name     = name,
                       basename = basename,
                       version  = version,
                       PATH     = PATH)

    write_channel_list(datafile,
                       PATH     = PATH,
                       basename = basename,
                       version  = version)

    if move:
        if '.' in name:
            name = name.split('.')[0]
        os.system('mv *mat *h5 params/{0}/'.format(name))


class ModelComparison(object):
    """
    ModelComparison takes the data from given runs and calculates
    various useful metrics including the most improved model, and
    the model with the lowest cost at the end of the run. The
    training data and the validation data are considered separately.

    Parameters
    ----------

        get_loss_specs: `callable`
            reads the mat files and updates the class with the loss info

        most_improved: `callable`
            returns a dict of the most improved model (in terms of loss) for
            training and testing

        lowest_cost: `callable`
            returns a dict of the model with the lowest for each training and
            testing
    """

    def __init__(self, data_dict):
        """
        Parameters
        ----------
            data_dict: `dict`
                dictionary containing {model1: mat_file1, model2: mat_file2 ...}
        """

        self.data_dict  = data_dict
        self.loss_specs = {}

    def get_loss_specs(self):
        """
        get_loss_specs updates the ModelComparison class with loss info
        for each model

        Parameters
        ----------
        data_dict: `dict`
            dictionary containing {model1: mat_file1, model2: mat_file2 ...}
        """

        for model, data in self.data_dict.items():
            data = sio.loadmat(data[0])

            # get start/end loss data
            train_loss_start = data['history'][0][0][0][0][0]
            train_loss_end   = data['history'][0][0][0][0][-1]
            val_loss_start   = data['history'][0][0][1][0][0]
            val_loss_end     = data['history'][0][0][1][0][-1]

            # save to dict
            self.loss_specs[model]                     = {}
            self.loss_specs[model]['train_loss_start'] = train_loss_start
            self.loss_specs[model]['train_loss_end']   = train_loss_end
            self.loss_specs[model]['val_loss_start']   = val_loss_start
            self.loss_specs[model]['val_loss_end']     = val_loss_end

    def most_improved(self):
        '''
        most_improved is a ModelComparison class method which calculates the
        model which has improved the most (in terms of loss) during training
        and the model which has improved the most during testing and gives this
        number as a percentage

        Returns
        -------
            dict: `dict`
                returns a dictionary containing the most improved
                test and train data
        '''
        max_dtrain = 1
        max_dtest  = 1
        most_improved_train = {}
        most_improved_test  = {}

        for model in self.loss_specs.keys():
            dtrain = self.loss_specs[model]['train_loss_end'] /\
                     self.loss_specs[model]['train_loss_start']

            dtest  = self.loss_specs[model]['val_loss_end'] /\
                     self.loss_specs[model]['val_loss_start']

            if 0 < dtrain < max_dtrain:
                most_improved_train = {'model': model, 'dtrain': dtrain * 100}
                max_dtrain = dtrain

            if 0 < dtest < max_dtest:
                most_improved_test = {'model': model, 'dtest': dtest * 100}
                max_dtest = dtest

        return {'train': most_improved_train, 'test': most_improved_test}

    def lowest_cost(self):
        '''
        lowest_cost is a ModelComparison class method which calculates the
        model which has the lowest cost for training and testing

        Returns
        -------
            dict: `dict`
                returns a dictionary of the model with the lowest cost at the
                end or training and the model with the lowest cost at the end
                of testing
        '''
        lowest_train_cost = {}
        lowest_test_cost  = {}
        First             = True

        for model in self.loss_specs.keys():
            end_train_cost = self.loss_specs[model]['train_loss_end']
            end_test_cost  = self.loss_specs[model]['val_loss_end']

            if First:
                train_cost = end_train_cost
                test_cost  = end_test_cost

                lowest_train_cost = {'model': model,
                                     'end_train_cost': end_train_cost}

                lowest_test_cost  = {'model': model,
                                     'end_test_cost': end_test_cost}
                First = False

            else:
                if end_train_cost < train_cost:
                    train_cost = end_train_cost

                    lowest_train_cost = {'model': model,
                                         'end_train_cost': end_train_cost}

                if end_test_cost < test_cost:
                    test_cost = end_test_cost

                    lowest_test_cost  = {'model': model,
                                         'end_test_cost': end_test_cost}


        return {'train': lowest_train_cost, 'test': lowest_test_cost}

    def plot_losses(self, plotDir="."):
        '''
        plot_losses plots the initial and final losses for both training
        and testing on a pie chart

        Parameters
        ----------
            plotDir: `string`
                directory to where the plots will be saved
        '''
        # set up colors and pie chart data
        color_dict = {'CNN': '#3b528b',
                      'GRU': '#21918c',
                      'LSTM': '#90c4e9',
                      'MLP': '#5ec962',
                      'bilinearRegression': '#90c4e9',
                      'bilinearRegressionReal': '#fde725',
                      'RNN_bilin': '#5ec962'}

        train_size         = []
        test_size          = []
        colors             = []
        labels             = []
        train_scale_factor = 0  # normalize to 1 for pie chart
        test_scale_factor  = 0  # normalize to 1 for pie chart

        for model in self.data_dict.keys():
            if model not in color_dict.keys():
                color_dict[model] = 'xkcd:steel blue'
            colors.append(color_dict[model])
            labels.append(model + str(self.data_dict[model][1]))

            train_loss = self.loss_specs[model]['train_loss_end']
            test_loss  = self.loss_specs[model]['val_loss_end']

            train_size.append(train_loss)
            train_scale_factor += train_loss

            test_size.append(test_loss)
            test_scale_factor += test_loss

        train_size /= train_scale_factor
        test_size  /= test_scale_factor

        # plot training losses
        train_fig, train_ax = plt.subplots()
        train_ax.pie(train_size,
                     labels     = labels,
                     autopct    = '%1.1f%%',
                     shadow     = True,
                     startangle = 90,
                     colors     = colors)
        plt.title('Relative Training Losses')
        train_ax.axis('equal')
        plt.savefig(plotDir + '/{}-training-losses.png'.format('-'.join(sorted(labels))))
        plt.savefig(plotDir + '/{}-training-losses.pdf'.format('-'.join(sorted(labels))))

        # plot validation losses
        test_fig, test_ax = plt.subplots()
        test_ax.pie(test_size,
                    labels     = labels,
                    autopct    = '%1.1f%%',
                    shadow     = True,
                    startangle = 90,
                    colors     = colors)
        plt.title('Relative Validation Losses')
        test_ax.axis('equal')
        plt.savefig(plotDir + '/{}-testing-losses.png'.format('-'.join(sorted(labels))))
        plt.savefig(plotDir + '/{}-testing-losses.pdf'.format('-'.join(sorted(labels))))


def update_links(path_to_html):
    '''
    update_links is meant to repopulate the dropdown menu for each model
    comparison html page with all of the hrefs. Without this, a particular
    run only 'knows' about the runs which came before it and cannot link to
    any runs that come after.

    Parameters
    ----------
        path_to_html: `string`
            path to where the html files are located. In general it will
            be something like `NonlinearRegression/HTML/{{ datestrymd }}.`
    '''

    # find all relevant files
    files      = os.listdir(path_to_html)
    raw_files  = [f for f in files if f.endswith('.html')
                                   if not f.startswith('index')]
    html_files = [path_to_html + html_file for html_file in raw_files]

    # collect every link
    all_links = []
    for html_file in html_files:

        if os.stat(html_file).st_size == 0:
            warning = 'WARNING: {}\nis empty. Skipping...'.format(html_file)
            wrapper = textwrap.TextWrapper(width=100)
            warning = wrapper.fill(text=warning)
            print(warning)
            continue

        with open(html_file) as html:
            soup   = BeautifulSoup(html, 'html.parser')
            ul     = soup.find('ul', {'class' : 'dropdown-menu'})
            links  = ul.find_all('li')
            prefix = '/'.join(path_to_html.split('/')[-2:]) + '/'
            run    = html_file.split('/')[-1]

            for l in links:
                all_links.append(l)

    all_links = sorted(list(set(all_links)))
    all_links = [al for al in all_links if not str(al).endswith('></a></li>')]

    # write every link to every file
    for html_file in html_files:

        if os.stat(html_file).st_size == 0:
            continue

        with open(html_file) as html:
            soup   = BeautifulSoup(html, 'html.parser')
            ul     = soup.find('ul', {'class' : 'dropdown-menu'})
            links  = ul.find_all('li')

            # Empty out unordered list section
            for li in links:
                li.decompose()

            prefix = '/'.join(path_to_html.split('/')[-2:]) + '/'

            # populate the section with all links
            for new_link in all_links:
                if new_link not in links:
                    ul.insert(0, '\n')
                    ul.insert(1, new_link)

        # save the changes
        updated_file = soup.prettify("utf-8")
        with open(html_file, "w") as f:
            f.write(str(updated_file))


def plot_loss(history, plotDir='.'):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.savefig('{}/Loss_history.png'.format(plotDir))
    plt.savefig('{}/Loss_history.pdf'.format(plotDir))
    plt.close()


def plot_timeseries(target, prediction, plotDir='.'):
    plt.plot(list(range(len(target))), target, label='target')
    plt.plot(list(range(len(prediction))), prediction, label='prediction')
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel('Time (s)')
    plt.savefig('{}/Timeseries_comparison.png'.format(plotDir))
    plt.savefig('{}/Timeseries_comparison.pdf'.format(plotDir))
    plt.close()


def plot_psd(target, prediction, bg, fs, nfft,
             plotDir    = '.',
             title_str  = None,
             basename   = None,
             version    = '0'):

    plotName = 'TF-bilinear-validation-{0}'.format(version)
    residual = target - prediction

    ff1, pp = sig.welch([target, prediction, residual, bg],
                        fs      = fs,
                        nperseg = nfft,
                        axis    = -1)

    strain = np.sqrt(pp).T

    # Lowpass filter the subtracted data
    temp = strain[:, 2].reshape(strain[:, 2].shape[0], 1)
    temp = ppr.filter_channels(temp, btype='low')
    strain[:, 2] = temp.reshape(len(temp))

    ff2, co = sig.coherence(target, prediction,
                            fs      = fs,
                            nperseg = nfft)

    title_str = 'Neural Network PSD Validation'
    info_str  = 'Model: {}'.format(basename)

    # make plots to evaluate success / failure of the regression
    fig, (ax1, ax2, ax3) = plt.subplots(nrows       = 3,
                                        sharex      = True,
                                        figsize     = (6, 8),
                                        gridspec_kw = {'height_ratios': [2, 1, 1]})

    ax1.set_prop_cycle(cycler('color', ['xkcd:jade',
                                        'xkcd:blood orange',
                                        'xkcd:burple',
                                        'xkcd:cement']))

    labels = ['DARM', 'Prediction', 'Subtracted']
    for i in range(3):
        ax1.loglog(ff1.T, strain[:,i],
                   alpha      = 0.8,
                   rasterized = True,
                   label      = labels[i])

    ax1.legend(fontsize='small', loc=1)
    ax1.set_xlim([7, 150])
    # ax1.set_ylim([5e-13, 1e-9])
    ax1.grid(True, which='minor')
    ax1.set_ylabel(r'DARM (m/$\sqrt{\rm Hz}$)')
    ax1.set_title(title_str)

    # Plot the ratio of the PSDs. Max ratio where DARM = subtracted
    ax2.semilogx(ff1.T, strain[:, 2] / strain[:, 0],
                 label      = 'DARM/Cleaned',
                 c          = 'xkcd:denim',
                 rasterized = True)

    ax2.loglog(ff1.T, [1 for _ in range(len(ff1.T))],
               label      = 'No Subtraction',
               c          = 'xkcd:black',
               linestyle  = ':',
               rasterized = True)

    ax2.grid(True, which='minor')
    ax2.set_ylim([1e-2, 50])
    ax2.set_ylabel(r'PSD Ratio')
    ax2.legend(fontsize='x-small', loc=1)

    # Plot the coherence
    ax3.semilogx(ff2.T, co,
                 label      = 'DARM/Prediction',
                 c          = 'xkcd:poop brown',
                 rasterized = True)

    ax3.grid(True, which='minor')
    ax3.set_ylim([0, 1.05])
    ax3.set_xlabel(r'Frequency (Hz)')
    ax3.set_ylabel(r'Coherence')

    ax3.legend(fontsize='x-small', loc=1)
    plt.subplots_adjust(hspace=0.075)

    # save figure,
    # backing up previous versions for easy comparison
    figName = '{0}/{1}'.format(plotDir, plotName)

    try:
        get_ipython
        plt.show()
    except NameError:
        plt.savefig('{}.png'.format(figName))
        plt.savefig('{}.pdf'.format(figName))


def write_optimizer_params(opt_name, optimizer,
                           basename = 'LSTM',
                           version  = '0',
                           PATH     = '.'):

    FILE = PATH + '/params/{0}/{0}-{1}.ini'.format(basename, version)

    configs = ConfigParser()
    configs.read(FILE)

    if opt_name == 'sgd':
       lr       = K.eval(optimizer.lr)
       decay    = K.eval(optimizer.decay)
       momentum = K.eval(optimizer.momentum)
       nesterov = str(optimizer.nesterov)

       args = [('lr',       lr),
               ('decay',    decay),
               ('momentum', momentum),
               ('nesterov', nesterov)]

    elif opt_name == 'adam' or opt_name == 'adamax' or opt_name == 'nadam':
       lr      = K.eval(optimizer.lr)
       decay   = K.eval(optimizer.decay)
       beta_1  = K.eval(optimizer.beta_1)
       beta_2  = K.eval(optimizer.beta_2)
       epsilon = float(optimizer.epsilon)

       args = [('lr',      lr),
               ('decay',   decay),
               ('beta_1',  beta_1),
               ('beta_2',  beta_2),
               ('epsilon', epsilon)]

    elif opt_name == 'rmsprop':
       lr      = K.eval(optimizer.lr)
       rho     = K.eval(optimizer.rho)
       decay   = K.eval(optimizer.decay)
       epsilon = float(optimizer.epsilon)

       args = [('lr',      lr),
               ('decay',   decay),
               ('rho',     rho),
               ('epsilon', epsilon)]

    elif opt_name == 'adadelta':
       lr      = K.eval(optimizer.lr)
       rho     = float(optimizer.rho)
       decay   = K.eval(optimizer.decay)
       epsilon = float(optimizer.epsilon)

       args = [('lr',      lr),
               ('decay',   decay),
               ('rho',     rho),
               ('epsilon', epsilon)]


    elif opt_name == 'adagrad':
       lr      = K.eval(optimizer.lr)
       decay   = K.eval(optimizer.decay)
       epsilon = float(optimizer.epsilon)

       args = [('lr',      lr),
               ('decay',   decay),
               ('epsilon', epsilon)]

    args.append(('name', opt_name))

    for tup in args:
        param, val = tup
        configs.set('optimizer', param, val)

    with open(FILE, 'w+') as configFile:
        configs.write(configFile)


def sort_optimizer_info(opt_tup_list):
    params = ['beta_1', 'beta_2', 'decay', 'epsilon',
              'lr', 'momentum', 'nesterov', 'rho']

    vals = [op[0] for op in opt_tup_list]
    for param in params:
        if param not in vals:
            opt_tup_list.append((param, '-'))

    return opt_tup_list


def split_dict(d, elements):
    """
    split_dict is used in plot_density in order to take a dict `d` of
    length `n` and split it into a list of dicts of length `elements`

    Parameters
    ----------
    d : `dict`
        input dictionary

    elements : `int`
        number of elements in each sub-dict

    Returns
    -------
    output : `list`
        list containing dicts of length `elements`
    """
    output = []
    chunks = int(len(d) / elements) + int(bool(len(d) % elements))
    for i in range(chunks):
        temp = {}
        if i < chunks - 1:
            for j in range(elements):
                temp[d.keys()[i * elements + j]] = d.values()[i * elements + j]
        else:
            remainder = len(d) - elements * i
            for j in range(remainder):
                temp[d.keys()[i * elements + j]] = d.values()[i * elements + j]

        output.append(temp)

    return output


def plot_channel_correlations(datafile,
                              plotNum   = 4,
                              data_type = None,
                              seconds   = 15,
                              plotDir   = '.'):
    """
    plot_channel_correlations calculates comparisons between channels
    in order to show which channels may contain 'features' of DARM.

    Parameters
    ----------
    datafile : `str`
        mat file to analyze

    plotNum : `int`
        Number of plots per image

    data_type : `str`
        use either 'real' or 'mock' data

    seconds : `int`
        How many seconds of data to query. NOTE: using times longer
        than ~30 seconds start to take a really long time to compute.
        If possible, use <= 30 seconds unless you're particularly
        patient :)

    plotDir : `str`
        path to store plots
    """
    # Get the datafile and type
    if data_type == None:
        if "data_array" in datafile:
            data_type = "real"
        elif "DARM_with" in datafile:
            data_type = "mock"

    # Get the data and separate darm from witnesses
    dataset, fs = ppr.get_dataset(datafile, data_type=data_type)

    # Scale it for nicer plotting
    scaler  = MinMaxScaler(feature_range=(0,1))
    dataset = scaler.fit_transform(dataset)

    # Cut it down to a reasonable size (1 minute)
    duration = int(fs * seconds)
    temp     = np.zeros(shape=(duration, dataset.shape[1]))

    for i in range(dataset.shape[1]):
        temp[:, i] = dataset[:duration, i]

    darm    = temp[:, 0]
    witness = temp[:, 2:]

    # Collect the channels or make them up
    if data_type == 'real':
        chans = sio.loadmat(datafile)['chans']
        chans = [c for c in chans if not "DARM" in c]
    else:
        chans = []
        for i in range(witness.shape[1]):
            chans.append("Witness {}".format(i + 1))

    data = dict(zip(chans, witness.T))
    data = split_dict(data, plotNum - 1)  # Not including DARM

    # Make sure the output directory exists
    if not os.path.isdir(plotDir):
        os.makedirs(plotDir)

    for index, chunk in enumerate(data):
        sys.stdout.write('\rMaking plot {0}/{1}'.format(index + 1, len(data)))
        sys.stdout.flush()
        if index == len(data) - 1:
            print('')

        # Add DARM and convert to DataFrame object
        chunk["DARM"] = darm
        df = pd.DataFrame(chunk)

        # Make the plots
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

        plt.suptitle('{0} Data Channel Correlations'.format(data_type.title()))
        plt.savefig('{0}/{1}_channel_corr_{2}.png'.format(plotDir, data_type, index))
        plt.savefig('{0}/{1}_channel_corr_{2}.pdf'.format(plotDir, data_type, index))
        plt.close()


def track_subtraction_progress(matfile, subtracted, target):
    if not os.path.isfile(matfile):
        os.system('touch {}'.format(matfile))

        output = {}
        output['darm_0'] = subtracted
        output['target'] = target
        sio.savemat(matfile, output)

    else:
        data = sio.loadmat(matfile)
        keys = [x for x in data.keys() if "darm" in x]
        max_version = max([int(list(x)[-1]) for x in keys]) + 1

        output = {}
        for key in keys:
            output[key] = data[key]
            output['target'] = data['target']

        darm_version = 'darm_{}'.format(max_version)
        output[darm_version] = subtracted

        sio.savemat(matfile, output)


def plot_progress(matfile, fs, nfft,
                  plotDir    = '.',
                  title_str  = None,
                  basename   = None,
                  version    = '0'):

    d = sio.loadmat(matfile)
    darms = [x for x in d.keys() if 'darm' in x]
    tar = np.array(d['target']).T
    tar = tar.reshape(len(tar))

    output = []
    output.append(tar)
    for k in darms:
        data = np.array(d[k].T)
        output.append(data.reshape(len(data)))

    shortest = min([output[i].shape[0] for i in range(len(output))])
    shortout = [output[i][-shortest:] for i in range(len(output))]
    plotName = 'Subtraction_Progress-{0}'.format(version)

    ff1, pp = sig.welch(shortout,
                        fs      = fs,
                        nperseg = nfft,
                        axis    = -1)
    strain  = np.sqrt(pp).T

    fig, ax1 = plt.subplots(nrows=1)
    ax1.set_prop_cycle(cycler('color', ['xkcd:black',
                                        'xkcd:blood orange',
                                        'xkcd:burple',
                                        'xkcd:cement']))

    labels = ['DARM', 'Calibration', 'Linear', 'Nonlinear']
    styles = ['-', '--', '-.', ':']
    for i in range(strain.shape[1]):
        ax1.loglog(ff1.T, strain[:,i],
                   alpha      = 0.8,
                   rasterized = True,
                   linestyle  = styles[i],
                   label      = labels[i])

    title_str = 'Subtraction Progress'
    ax1.legend(fontsize='small')
    ax1.set_xlim([7, 30])
    ax1.set_ylim([5e-13, 5e-10])
    ax1.grid(True, which='minor')
    ax1.set_ylabel(r'DARM (m/$\sqrt{\rm Hz}$)')
    ax1.set_xlabel(r'Frequency (Hz)')
    ax1.set_title(title_str)

    # Save figure
    figName = '{0}/{1}'.format(plotDir, plotName)

    try:
        get_ipython
        plt.show()
    except NameError:
        plt.savefig('{}.png'.format(figName))
        plt.savefig('{}.pdf'.format(figName))
