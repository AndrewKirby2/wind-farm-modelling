"""1. Calculate Cp from LES data for a finite
wind farm
2. Predict Cp using analytical model of the local
turbine thrust coefficient
3. Predict Cp using statistical model of the
local turbine thrust coefficient
"""

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

#wind farm parameters
#momentum `extractability' factor
zeta=20
#bottom friction exponent
gamma=2

#############################################
# 1. Calculate Cp from LES data for a finite
# wind farm
#############################################

#arrays to store result
cp_finite = np.zeros(50)
effective_area_ratio = np.zeros(50)

#load LES data
training_data = np.genfromtxt('LES_training_data.csv', delimiter=',')
#remove header
training_data = np.delete(training_data, 0, 0)
training_data = np.delete(training_data, 0, 1)

beta = training_data[:,5]
cp = training_data[:,7]

#calculate adjusted Cp and effective area ratio
#for each wind farm LES
for run_no in range(50):
    U_F = beta[run_no]*10.10348311
    U_F0 = 10.10348311

    #coefficients of quadratic formula to solve
    a = 1/U_F**2
    b = zeta/U_F0
    c = -zeta - 1

    U_Fprime = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    
    cp_finite[run_no] = cp[run_no]*(U_Fprime/U_F)**3

    #calculate effective area ratio
    C_f0 = 0.28641758**2/(0.5*10.10348311**2)
    A = np.pi/4
    S = training_data[run_no,0]*training_data[run_no,1]
    area_ratio = A/S
    effective_area_ratio[run_no] = area_ratio/C_f0

#############################################
# 2. Predict Cp using analytical model of the local
turbine thrust coefficient
#############################################

cp_analytical = np.zeros(50)

#predict Cp for each wind farm LES
for i in range(50):

    def NDFM(beta):
        """ Non-dimensional farm momentum
        equation (see Nishino 2020)
        """
	#use uncorrected ct_star to predict beta
	#analytical model gives ct_star = 0.75
	#divide by correction factor (N^2=0.8037111)
	ct_star_adj = 0.75 / 0.8037111
        return ct_star_adj*effective_area_ratio[i]*beta**2 + beta**gamma - 1 -zeta*(1-beta)

    beta = sp.bisect(NDFM,0,1)
    cp_analytical[i] = 0.75**1.5 * beta**3 * 1.33**-0.5

#############################################
# 3. Predict Cp using statistical model of the local
turbine thrust coefficient
#############################################

#load statistical model predictions for ct_star
#made using LOOCV
ct_star_statistical = np.genfromtxt('local_turbine_thrust_coefficient_predictions.csv', delimiter=',')[1:,2]

cp_statistical = np.zeros(50)

#predict Cp for each wind farm LES
for i in range(50):

    def NDFM(beta):
        """ Non-dimensional farm momentum
        equation (see Nishino 2016)
        """
	#use uncorrected ct_star to predict beta
	#analytical model gives ct_star = 0.75
	#divide by correction factor (N^2=0.8037111)
	ct_star_adj = ct_star_statistical[run_no] / 0.8037111
        return ct_star_adj*effective_area_ratio[i]*beta**2 + beta**gamma - 1 -zeta*(1-beta)

    beta = sp.bisect(NDFM,0,1)
    cp_statistical[i] = ct_star_statistical[i]**1.5 * beta**3 * 1.33**-0.5


print('Mean absolute percentage error for Cp predictions (using analytical model) = ',100*mean_absolute_percentage_error(cp_finite, cp_analytical))
print('Mean absolute percentage error for Cp predictions (using statistical model) = ',100*mean_absolute_percentage_error(cp_finite, cp_statistical))
