"""1. Calculate Cp from LES data different
wind farm "extractability"
2. Predict Cp using two scale momentum theory
3. Plot results
"""

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

#wind farm parameters
#momentum `extractability' factor
zeta=[0,5,10,15,20,25]
#bottom friction exponent
gamma=2

#arrays to store result
cp_finite = np.zeros((50,6))
effective_area_ratio = np.zeros(50)
cp_nishino = np.zeros((50,6))

#load LES data
training_data = np.genfromtxt('LES_training_data.csv', delimiter=',')
#remove header
training_data = np.delete(training_data, 0, 0)
training_data = np.delete(training_data, 0, 1)

beta = training_data[:,5]
cp = training_data[:,7]

#repeat for different zeta values
for i in range(6):

    #############################################
    # 1. Calculate Cp from LES data for a finite
    # wind farm
    #############################################

    #calculate adjusted Cp and effective area ratio
    #for each wind farm LES
    for run_no in range(50):
        U_F = beta[run_no]*10.10348311
        U_F0 = 10.10348311

        #coefficients of quadratic formula to solve
        a = 1/U_F**2
        b = zeta[i]/U_F0
        c = -zeta[i] - 1

        U_Fprime = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    
        cp_les[run_no,i] = cp[run_no]*(U_Fprime/U_F)**3

        #calculate effective area ratio
        C_f0 = 0.28641758**2/(0.5*10.10348311**2)
        A = np.pi/4
        S = training_data[run_no,0]*training_data[run_no,1]
        area_ratio = A/S
        effective_area_ratio[run_no] = area_ratio/C_f0

    #############################################
    # 2. Predict Cp using two-scale momentum
    # theory
    #############################################
	
    effective_area_ratio_theory = np.linspace(0,20,50)

    #predict Cp for each wind farm LES
    for run_no in range(50):

        def NDFM(beta):
            """ Non-dimensional farm momentum
            equation (see Nishino 2020)
            """
	    #use uncorrected ct_star to predict beta
	    #analytical model gives ct_star = 0.75
	    #divide by correction factor (N^2=0.8037111)
	    ct_star_adj = 0.75 / 0.8037111
            return ct_star_adj*effective_area_ratio_theory[i]*beta**2 + beta**gamma - 1 -zeta[i]*(1-beta)

        beta = sp.bisect(NDFM,0,1)
        cp_nishino[run_no,i] = 0.75**1.5 * beta**3 * 1.33**-0.5

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[5.33,6.6], dpi=600)
ax[0,0].plot(effective_area_ratio_theory[12:], cp_theory[12:,0])
pcm = ax[0,0].scatter(effective_area_ratio, training_data[:,8], s=5, c=training_data[:,3])
ax[0,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[0,0].set_ylabel(r'$C_p$')
ax[0,0].set_title('a)', loc='left')

ax[0,1].plot(effective_area_ratio_theory[12:], cp_theory[12:,1])
ax[0,1].scatter(effective_area_ratio, cp_finite[:,1], s=5, c=training_data[:,3])
ax[0,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[0,1].set_ylabel(r'$C_{p,finite}$')
ax[0,1].set_title('b)', loc='left')

ax[1,0].plot(effective_area_ratio_theory[12:], cp_theory[12:,2])
ax[1,0].scatter(effective_area_ratio, cp_finite[:,2], s=5, c=training_data[:,3])
ax[1,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,0].set_ylabel(r'$C_{p.finite}$')
ax[1,0].set_title('c)', loc='left')

ax[1,1].plot(effective_area_ratio_theory[12:], cp_theory[12:,3])
ax[1,1].scatter(effective_area_ratio, cp_finite[:,3], s=5, c=training_data[:,3])
ax[1,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,1].set_ylabel(r'$C_{p,finite}$')
ax[1,1].set_title('d)', loc='left')

ax[2,0].plot(effective_area_ratio_theory[12:], cp_theory[12:,4])
ax[2,0].scatter(effective_area_ratio, cp_finite[:,4], s=5, c=training_data[:,3])
ax[2,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,0].set_ylabel(r'$C_{p,finite}$')
ax[2,0].set_title('e)', loc='left')

ax[2,1].plot(effective_area_ratio_theory[12:], cp_theory[12:,5])
ax[2,1].scatter(effective_area_ratio, cp_finite[:,5], s=5, c=training_data[:,3])
ax[2,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,1].set_ylabel(r'$C_{p,finite}$')
ax[2,1].set_title('f)', loc='left')

plt.tight_layout()

cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), shrink=0.97)
cbar.set_label(r'$C_T^*$')

plt.savefig('LES_cp_results.png', bbox_inches='tight')

