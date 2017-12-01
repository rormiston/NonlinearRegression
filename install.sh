#########
# Intro #
#########
echo "This script will install NonlinearRegression into a virtual environment.
Make sure that you are not already sourced in another environment. The VE will
be created and installed to $HOME/noise_cancellation. Additionally, MockData,
pynoisesub, PyMultiNest (and MultiNest) will be installed here unless you
already have them."
echo ""

echo -n "Do you wish to proceed? (y/n): "
read proceed

if [ "$proceed" = "n" ]; then
    echo "exiting..."
    exit
else
    echo "starting install..."
    basedir=$(pwd)
fi


##################################
# Create the virtual environment #
##################################
cd $HOME

# Make sure virtualenv exists
if hash virtualenv 2> /dev/null; then
    virtualenv noise_cancellation
else
    echo "Need to install virtualenv"
    echo -n "Install it now? (y/n): "
    read install

    if [ "$install" = "n" ]; then
        echo "exiting..."
        exit
    else
        echo "installing virtualenv..."
        pip install virtualenv
        virtualenv noise_cancellation
    fi
fi

# source the VE
source $HOME/noise_cancellation/bin/activate
pip install numpy

#####################################
# Install the repo and dependencies #
#####################################
cd $basedir

# Install NonlinearRegression
pip install --upgrade pip
pip install -r requirements.txt && pip install .

# Check for PyMultiNest and install if necessary
MultiNest=$LD_LIBRARY_PATH

if [[ $MultiNest == *"MultiNest"* ]]; then
    echo "MultiNest already installed. Skipping"
else
    cd ..
    repo_dir=$(pwd)

    # First install multinest
    echo "Installing MultiNest..."
    git clone https://github.com/JohannesBuchner/MultiNest.git
    cd MultiNest/build/
    cmake .. && make
    export LD_LIBRARY_PATH=$repo_dir/MultiNest/lib/:$LD_LIBRARY_PATH

    # Next install PyMultiNest
    cd $repo_dir
    git clone https://github.com/JohannesBuchner/PyMultiNest.git
    cd PyMultiNest
    python setup.py install
fi

# Install MockData and pynoisesub if we need to
cd $repo_dir

mock_dir=$repo_dir/MockData
if [ ! -d "$mock_dir" ]; then
    echo "installing MockData..."
    git clone git@git.ligo.org:NoiseCancellation/MockData.git
fi

pynoisesub_dir=$repo_dir/pynoisesub
if [ ! -d "$pynoisesub_dir" ]; then
    echo "installing pynoisesub..."
    git clone git@git.ligo.org:NoiseCancellation/pynoisesub.git
    cd pynoisesub
    pip install .
fi

# For some reason, the html templates and shared libraries
# don't like to install, so let's do it manually
cd $basedir
cp NonlinearRegression/templates/* $HOME/noise_cancellation/lib/python2.7/site-packages/NonlinearRegression/templates/
cp NonlinearRegression/tools/*.so $HOME/noise_cancellation/lib/python2.7/site-packages/NonlinearRegression/tools/

#########################
# Write the config file #
#########################
mkdir -p $HOME/noise_cancellation/configs/
touch $HOME/noise_cancellation/configs/configs.ini

cat <<EOF > $HOME/noise_cancellation/configs/configs.ini
[run]
basedir = $(pwd)
models  =
EOF

###################################
# Test imports and relative paths #
###################################
cd $basedir
echo "
*******************************************************************************
"
python -B tests/get_imports.py -b $basedir
python -B tests/test_imports.py

#############
# Finish up #
#############
echo "
Remember to source the virtual environment before running. i.e.,

source $HOME/noise_cancellation/bin/activate

Your config file is in $HOME/noise_cancellation/configs and needs to
be edited before running model comparisons. Check the README for
further usage instructions.
Done"
