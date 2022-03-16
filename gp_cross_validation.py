"""Perform Leave-One-Out Cross-Validation to 
estimate test error of statistical model
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

#load input parameters (S_x, S_y, theta)
X = training_data[:,:3]
#load output C_T^*
y = training_data[:,3]

#load wake model prior mean
prior_mean = np.genfromtxt('local_turbine_thrust_ceofficient_predictions.csv', delimiter=',')[:,0]

ctstar_statistical_model = np.zeros(50)
prediction_error = np.zeros(50)

loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(test_index)

    #standardise the feature set of the training and test data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_stan = scaler.transform(X_train)


    #create kernel for Gaussian Process Regression
    kernel = 1.0 ** 2 * RBF(length_scale=[1.,1.,1.]) + WhiteKernel(noise_level=1e-3, noise_level_bounds=[1e-10,1])
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)


    #fit GP and make predictions
    gp.fit(X_train_stan,y_train-prior_mean[train_index])
    print(gp.kernel_)

    X_test_stan = scaler.transform(X_test)

    #print(mean_absolute_error(y_test, gp.predict(X_test_stan) + 
     # prior_mean[test_index])/0.75)
    error[test_index] = mean_absolute_error(y_test, gp.predict(X_test_stan) + 
      prior_mean[test_index])/0.75
    ctstar[test_index] = gp.predict(X_test_stan) + prior_mean[test_index]
    

np.savetxt('ct_star_statistical_model.csv', ctstar)

print(np.mean(error))
print(np.max(error))

np.save('ctstar_nishino_error', 100*np.sort(np.abs(y_train-0.75)/0.75))
np.save('ctstar_statistical_model_error', 100*np.sort(error))

plt.plot(100*np.sort(np.abs(y_train-0.75)/0.75), label='$C_T^*=0.75$')
plt.plot(100*np.sort(error), label=r'$C_T^*=f(S_x, S_y, \theta)$')
plt.legend()
plt.ylabel('Prediction error (%)')
plt.xlabel('Sorted testing points')
plt.savefig('prediction_error.png')