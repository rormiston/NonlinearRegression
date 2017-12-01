.. NonlinearRegression documentation master file, created by
   sphinx-quickstart on Tue Sep 12 12:09:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NonlinearRegression's documentation!
===============================================
This repository contains several separate approaches to nonlinear regression,
mostly using TensorFlow. The most successful/thoroughly commented one can be
run with ``tensorflowExample/lattice/tf_wit2d.py``, which will generate several
plots including ASDs of the noise, input and resulting data.

Other examples can be run with ``tensorflowExample/vae/variable_autoencoder.py``,
``tensorflowExample/tf.py``, and ``matlabNN/fitScat.m``. These examples expect data
files which can be generated using scripts in the root directory of the
MockData repository: ``make_STFT_data.py`` for the TensorFlow examples and
``makeNoise.py`` for the Matlab example.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial/tutorial
   tools/tools



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
