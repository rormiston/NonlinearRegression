=========
Tutorials
=========

The sections below give a quick overview of the installation and
how to use some of the  tools within the NonlinearRegression package

Installation
------------
Installing the NonlinearRegression repository is a simple task. First, 
clone the repository.

.. code-block:: bash

   $ git clone git@git.ligo.org:rich.ormiston/NonlinearRegression.git

Next, cd to the base directory and run the install script.

.. code-block:: bash

   $ cd NonlinearRegression
   $ chmod +x install.sh
   $ ./install.sh

And that's it! The virtual environment will be created for you and
all of the dependencies will be installed within it. NOTE: Before
the scipts may be run, you need to source the new environment

.. code-block:: bash

    $ source $HOME/noise_cancellation/bin/activate


Running Scripts
---------------
Assuming that the repository install correctly and that data has been
generated from the MockData repository, then the scripts should work
"out-of-the-box." For example, the following will iterate over 15 epochs,

.. code-block:: bash

   $ python RNN_bilin.py -e 15

Generated data will be stored in the ``params/`` directory under the 
appropriate model name.

Model Comparison
----------------
When running multiple models over and over, it is necessary to have an
efficient way of comparing their performance. To do this, first the 
configuration file must be edited to include the models that are to be
compared. This file is found under ``$HOME/noise_cancellation/configs/``.
The model names are simply the script names without the extension. 
For example, a config file may look as follows:

.. code-block:: ini

   [run]
   basedir = albert.einstein/git_repositories/NonlinearRegression
   models  = RNN_bilin, RCNN_bilin, bilinearRegression

The model names can be separated by any non-word characters (in regex terms,
anything that would not be \\w+, but rather \\W+).

After the config has been edited, the webpages may be easily generated
by running the ``nlr-compare-models`` command

.. code-block:: bash

   $ nlr-compare-models -i path/to/config/file

By default the config file path is set to ``$HOME/noise_cancellation/configs``
so if you have not moved this file, no flag is necessary. If all went well,
you should see a message that the webpages were built successfully. They are
stored in ``NonlinearRegression/NonlinearRegression/HTML/<YYYYMMDD>`` and navigating
a web browser to this page should show the results. A dropdown bar called "Runs"
will update as more comparisons are run that day. The calendar is available to
jump to a different run date. **A demo of the sample output can be found** here_

.. _here: https://ldas-jobs.ligo-la.caltech.edu/~rich.ormiston/NonlinearRegression/NonlinearRegression/HTML/20170908/index.html


=============
Common Errors
=============
There have been some quirks upon installation and during usage and they are
noted below. None are serious issues, just annoying. If a problem other than
one shown below occurs, please email ``rich.ormiston@ligo.org`` with the issues. 

MultiNest library path
----------------------
It has sometimes been seen that, while MultiNest install seccessfully, the 
LD_LIBRARY_PATH is not set during the install and therefore ``optswarm.py``
and ``swarmRegression.py`` may fail with the error:

.. code-block:: bash

   $ ERROR: Could not load MultiNest library "libmultinest.so"

If this occurs, simply enter the following to re-export the path library path
variable:

.. code-block:: bash 

   $ export LD_LIBRARY_PATH=path_to_repositories/MultiNest/lib/:$LD_LIBRARY_PATH   


Color key not found while generating webpages
---------------------------------------------
If a new script was added, or the name changed, then the webpage build will
fail due to a key error about a missing color. In ``tools/analyze_run_data.py``
there is a color dict in the method ``plot_losses`` of the
:py:class:`ModelComparison` class. Each script is given a color for the sake of
consistency across runs. To correct this error, simply add a new color dict item.
Remember, you must either change the file in your site-packages diretory or, if
you changed the file in the repository itself, you must reinstall the package for
the changes to take effect. This is done by going to the top level directory (the
one with the ``setup.py`` file and running

.. code-block:: bash

   $ pip install .


RCNN_bilin will not run
-----------------------
If ``RCNN_bilin.py`` failed, it is almost certainly due to the version of tensorflow
that is installed - namely, it only runs on CPUs but ``RCNN_bilin.py`` requires GPUs
in order to run. If you have available GPUs to run on, this is an easy fix. We can 
uninstall the current version of tensorflow and install the version with GPU support.
All of the other scripts will run with the GPU version, so you need not switch back
and forth. Do the following (and make sure you're sourced!)

.. code-block:: bash

   $ pip uninstall tensorflow
   $ pip install tensorflow-gpu


Webpage build fails with "optimizer" section not found
------------------------------------------------------
The most common reason for the webpage build to fail is that the config file
appears to be missing sections. The problem is not the config file itself but rahter
the *path* to the file. Most likely, user-specified directories were used upon install
and so the default path to the config file is incorrect. It is also possible that the
fields in the config file contain essential non-word characters which are excluded 
by the regex matching and therefore the filename is incorrect and so the script attempts
to read a file that does not exist. To be sure, add a print statement to the ``ini_file``
variable in ``bin/nlr-compare-models`` and verify that the config file that it is attempting
to read truly exists. 
