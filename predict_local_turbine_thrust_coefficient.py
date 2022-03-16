"""Perform Leave-One-Out Cross-Validation (LOOCV)
to estimate test error of statistical model
C_T^* = f(S_x, S_y, theta)
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut

#load the LES training data
training_data = np.genfromtxt('LES_training_data.csv', delimiter = ',')
#remove header
training_data = np.delete(training_data, 0, 0)
training_data = np.delete(training_data, 0, 1)

#load input parameters (S_x, S_y, theta)
X = training_data[:,:3]
#load output C_T^*
y = training_data[:,3]

#load wake model prior mean
prior_mean = np.genfromtxt('local_turbine_thrust_ceofficient_predictions.csv', delimiter=',')[1:,1]

ctstar_statistical_model = np.zeros(50)
prediction_error = np.zeros(50)

print('Test point   Mean Absolute Error (%)')

#perform LOOCV
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #standardise the feature set of the training and test data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_stan = scaler.transform(X_train)

    #create kernel for Gaussian Process Regression
    kernel = 1.0 ** 2 * RBF(length_scale=[1.,1.,1.]) + WhiteKernel(noise_level=1e-3, noise_level_bounds=[1e-10,1])
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)

    #fit Gaussian Process
    gp.fit(X_train_stan,y_train-prior_mean[train_index])

    #make prediction on data point left out of training set
    X_test_stan = scaler.transform(X_test)
    error[test_index] = mean_absolute_error(y_test, gp.predict(X_test_stan) + 
      prior_mean[test_index])/0.75
    ctstar_statistical_model[test_index] = gp.predict(X_test_stan) + prior_mean[test_index]
    
    print(test_index,'          ', 100*error[test_index])
