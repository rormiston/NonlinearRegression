"""
This is a simple wrapper for `train_network.py`. The purpose is to
loop over the network training process and use the prediction of each
pass as the target of the following iteration. The first pass uses data
made from only PCAL and DARM and aims to subtract the calibration lines.
"""
import sys
import re
import os
import NonlinearRegression.tools.analyze_run_data as ard


# Start message
iterate = 'Running Multi-Pass Subtraction'
print('\n{0}\n{1}'.format(iterate, '-' * len(iterate)))

# Make sure the data loads when running from any directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Collect the command line flags
flags = ' '.join(sys.argv[1:])

# Make sure a model is declared
model_type = re.compile(r'[a-zA-Z]+')
model_version = re.compile(r'\d+')
if not "-m" in flags:
    print("ERROR: Must specify a model using the -m flag")
    sys.exit(1)
else:
    flag_list = flags.split(' ')
    for flag in flag_list:
        if "-m" in flag:
            index = flag_list.index(flag)

model = flag_list[index + 1]
basename = model_type.findall(model)[0]
try:
    version = model_version.findall(model)[0]
except IndexError:
    version = '0'

# Delete the old results so we don't append to them
matfile = 'params/{0}/subtraction_results-{1}.mat'.format(basename, version)
if os.path.isfile(matfile):
    os.system('rm {}'.format(matfile))

# Loop three times - Once for calibration lines, once for linear couplings,
# and once more to remove as many nonlinear couplings as possible.
for ii in range(3):
    if ii == 0:
        print('[+] Removing calibration lines...')
        cal_file = ' --datafile Data/L1_calibration.mat'
        os.system('python train_network.py {}'.format(flags + cal_file))

    if ii == 1:
        print('\n[+] Removing first order noise sources...')
        os.system('python train_network.py {}'.format(flags + ' --doLoops'))

    if ii == 2:
        print('\n[+] Removing nonlinear sources...')
        os.system('python train_network.py {}'.format(flags + ' --doLoops'))

fs = 256
for flag in flag_list:
    if "--fs_slow" in flag:
        index = flag_list.index(flag)
        fs = int(flag_list[index + 1])

nperseg = fs * 8
plotDir = 'params/{0}/Figures'.format(basename)
ard.plot_progress(matfile, fs, nperseg, plotDir=plotDir, version=version)
