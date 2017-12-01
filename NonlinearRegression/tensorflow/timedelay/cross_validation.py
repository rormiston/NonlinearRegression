import matplotlib
matplotlib.use('agg')
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import NonlinearRegression.tools.preprocessing as ppr
import numpy as np
import os

# Hush TF AVX warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load dataset
dataset, _ = ppr.get_dataset('Data/L1_data_array.mat')
scaler     = StandardScaler()
dataset    = scaler.fit_transform(dataset)

X = dataset[:,1:]
Y = dataset[:, 0]

# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=X.shape[1], activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='linear'))

	# Compile model
	model.compile(loss      = 'mse',
                  optimizer = 'adam',
                  metrics   = ['accuracy'])
	return model

# create model
model = KerasClassifier(build_fn   = create_model,
                        epochs     = 150,
                        batch_size = 1000,
                        verbose    = 1)

# evaluate using 10-fold cross validation
kfold = KFold(n_splits     = 10,
              shuffle      = True,
              random_state = seed)

results = cross_val_score(model, X, y=Y, cv=kfold)
print(results.mean())
