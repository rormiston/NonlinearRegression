## Instructions
This repository contains several separate approaches to nonlinear regression,
mostly using TensorFlow. The most successful/thoroughly commented one can be
run with tensorflowExample/lattice/tf_wit2d.py, which will generate several
plots including ASDs of the noise, input and resulting data.

Other examples can be run with tensorflowExample/vae/variable_autoencoder.py,
tensorflowExample/tf.py, and matlabNN/fitScat.m. These examples expect data
files which can be generated using scripts in the root directory of the
MockData repository: make_STFT_data.py for the TensorFlow examples and
makeNoise.py for the Matlab example. 


## Installation
After cloning the repository, simply run
```shell
$ chmod +x install.sh
$ ./install.sh
```
and the package will install into a new virtual environment called
"noise\_cancellation." 


## Usage
* Modify `basedir` and `models` for the ini file located in `configs/model_comparison.ini`.
`basedir` is the path to the top level directory of the repository and `models` is the
list of models to compare. Make sure that the model names are the same as the script names
but without the extension. 

* Create the data for the models you want to compare and run the models.
NOTE: You may need to change the path to the data files!

* To finally comapre the model and make the webpage, simply run
```shell
$ nlr-compare-models -i path/to/ini/file
```
If you haven't changed the path to the config file which was set up by install.sh, then
the default path will work and the `-i` flag may be left off. The output will go to
`basedir/NonlinearRegression/HTML/yyyymmdd`. To view the webpage, make sure that
_at least_ the HTML directory is softlinked to your public\_html

## Reading
1. http://www.scholarpedia.org/article/Reinforcement_learning
1. https://azure.microsoft.com/en-us/documentation/services/machine-learning/
1. http://playground.tensorflow.org/
1. http://usblogs.pwc.com/emerging-technology/demystifying-machine-learning/


## Nonlinear Regression:
1. http://www.cs.cmu.edu/~zkolter/course/15-830-f12/ml_nonlin_reg.pdf
1. http://www.astroml.org/sklearn_tutorial/general_concepts.html
1. http://scikit-learn.org/stable/supervised_learning.html
1. http://www.mathworks.com/help/stats/regression-and-anova.html
1. https://www.cs.colostate.edu/~anderson/cs480/doku.php?id=schedule)
1. http://web.unbc.ca/~ytang/Chapter6.pdf


## Neural Networks
1. Intro to Neural Nets: http://csc.lsu.edu/~jianhua/nn.pdf


## General Machine Learning
1. http://deeplearning.net/tutorial/
1. http://www.mathworks.com/discovery/supervised-learning.html
1. http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/
1. TensorFlow tutorials: http://tensorflowtutorial.net/tensorflow-tutorial-resources


### Supervised Learning is better for Regression than Unsupervised or Reinforcement Learning
1. https://en.wikipedia.org/wiki/Supervised_learning


## Support Vector Machines:
1. https://en.wikipedia.org/wiki/Support_vector_machine


## Courses on Machine Learning
https://www.coursera.org/learn/ml-regression
