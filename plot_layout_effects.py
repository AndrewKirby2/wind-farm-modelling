"""1. Calculate Cp from LES data different
wind farm "extractability"
2. Predict Cp using two scale momentum theory
3. Plot results
"""

import numpy as np
import scipy.optimize as sp
import matplotlib.pyplot as plt

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

#note correction factor N^2 already applied!
ct_star = training_data[:,3]
beta = training_data[:,5]
#note correction factor N^3 already applied!
cp = training_data[:,7]
cp_corrected = np.zeros((50))
beta_corrected = np.zeros((50))

################################
#1. Correct LES wind speed
################################

for run_no in range(50):

        #calculate effective area ratio
        C_f0 = 0.28641758**2/(0.5*10.10348311**2)
        A = np.pi/4
        S = training_data[run_no,0]*training_data[run_no,1]
        area_ratio = A/S
        effective_area_ratio[run_no] = area_ratio/C_f0

        #calculate beta_fine_theory
        def NDFM(beta):
            """ Non-dimensional farm momentum
            equation (see Nishino 2020)
            """
	    #use ct_star to predict beta_fine_theory
            return ct_star[run_no]*effective_area_ratio[run_no]*beta**2 + beta**gamma - 1

        beta_fine_theory = sp.bisect(NDFM,0,1)

        #calculate beta_coarse_theory
        def NDFM(beta):
            """ Non-dimensional farm momentum
            equation (see Nishino 2020)
            """
	    #use ct_star to predict beta_fine_theory
            return (ct_star[run_no]/ 0.8037111)*effective_area_ratio[run_no]*beta**2 + beta**gamma - 1

        beta_coarse_theory = sp.bisect(NDFM,0,1)

        #correct Cp values recorded by LES
        cp_corrected[run_no] = cp[run_no]*(beta_fine_theory/beta_coarse_theory)**3
        #correct beta value recorded by LES
        beta_corrected[run_no] = beta[run_no]*(beta_fine_theory/beta_coarse_theory)


#repeat for different zeta values
for i in range(6):

    #############################################
    # 2. Calculate Cp from LES data for a finite
    # wind farm
    #############################################

    #calculate adjusted Cp and effective area ratio
    #for each wind farm LES
    for run_no in range(50):
        U_F = beta_corrected[run_no]*10.10348311
        U_F0 = 10.10348311

        #coefficients of quadratic formula to solve
        a = 1/U_F**2
        b = zeta[i]/U_F0
        c = -zeta[i] - 1

        U_Fprime = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    
        cp_finite[run_no,i] = cp_corrected[run_no]*(U_Fprime/U_F)**3

    #############################################
    # 3. Predict Cp using two-scale momentum
    # theory
    #############################################

    #predict Cp for each wind farm LES
    for run_no in range(50):

        def NDFM(beta):
            """ Non-dimensional farm momentum
            equation (see Nishino 2020)
            """
	    #use ct_star to predict beta
	    #analytical model gives ct_star = 0.75
            ct_star = 0.75
            return ct_star*effective_area_ratio[run_no]*beta**2 + beta**gamma - 1 -zeta[i]*(1-beta)

        beta_theory = sp.bisect(NDFM,0,1)
        cp_nishino[run_no,i] = 0.75**1.5 * beta_theory**3 * 1.33**-0.5

turbine_loss_factor = 1 - cp_finite/cp_nishino
farm_loss_factor = 1 - cp_nishino/0.563205
total_loss_factor = 1 - cp_finite/0.563204

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[5.33,6.6], dpi=600)
pcm = ax[0,0].scatter(effective_area_ratio, turbine_loss_factor[:,0], s=5, c=training_data[:,3])
ax[0,0].set_xlabel(r'$\lambda/C_{f0}$') 
ax[0,0].set_ylabel(r'$\Pi_T$')
ax[0,0].set_title('a)', loc='left')
ax[0,0].minorticks_on()
ax[0,0].set_ylim(-0.05,0.25)
ax[0,0].grid(which='major', axis='y', color='k')
ax[0,0].grid(which='minor', axis='y', linestyle='--')

ax[0,1].scatter(effective_area_ratio, turbine_loss_factor[:,1], s=5, c=training_data[:,3])
ax[0,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[0,1].set_ylabel(r'$\Pi_T$')
ax[0,1].set_title('b)', loc='left')
ax[0,1].minorticks_on()
ax[0,1].set_ylim(-0.05,0.25)
ax[0,1].grid(which='major', axis='y', color='k')
ax[0,1].grid(which='minor', axis='y', linestyle='--')

ax[1,0].scatter(effective_area_ratio, turbine_loss_factor[:,2], s=5, c=training_data[:,3])
ax[1,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,0].set_ylabel(r'$\Pi_T$')
ax[1,0].set_title('c)', loc='left')
ax[1,0].minorticks_on()
ax[1,0].set_ylim(-0.05,0.25)
ax[1,0].grid(which='major', axis='y', color='k')
ax[1,0].grid(which='minor', axis='y', linestyle='--')

ax[1,1].scatter(effective_area_ratio, turbine_loss_factor[:,3], s=5, c=training_data[:,3])
ax[1,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,1].set_ylabel(r'$\Pi_T$')
ax[1,1].set_title('d)', loc='left')
ax[1,1].minorticks_on()
ax[1,1].set_ylim(-0.05,0.25)
ax[1,1].grid(which='major', axis='y', color='k')
ax[1,1].grid(which='minor', axis='y', linestyle='--')

ax[2,0].scatter(effective_area_ratio, turbine_loss_factor[:,4], s=5, c=training_data[:,3])
ax[2,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,0].set_ylabel(r'$\Pi_T$')
ax[2,0].set_title('e)', loc='left')
ax[2,0].minorticks_on()
ax[2,0].set_ylim(-0.05,0.25)
ax[2,0].grid(which='major', axis='y', color='k')
ax[2,0].grid(which='minor', axis='y', linestyle='--')

ax[2,1].scatter(effective_area_ratio, turbine_loss_factor[:,5], s=5, c=training_data[:,3])
ax[2,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,1].set_ylabel(r'$\Pi_T$')
ax[2,1].set_title('f)', loc='left')
ax[2,1].minorticks_on()
ax[2,1].set_ylim(-0.05,0.25)
ax[2,1].grid(which='major', axis='y', color='k')
ax[2,1].grid(which='minor', axis='y', linestyle='--')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5)

cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), shrink=0.97)
cbar.set_label(r'$\overline{C_T^*}$')

plt.savefig('LES_turbine_loss_factor.png', bbox_inches='tight')

plt.close()

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=[5.33,6.6], dpi=600)
pcm = ax[0,0].scatter(effective_area_ratio, turbine_loss_factor[:,0]/farm_loss_factor[:,0], s=5, c=training_data[:,3])
ax[0,0].set_xlabel(r'$\lambda/C_{f0}$') 
ax[0,0].set_ylabel(r'$\Pi_T/\Pi_F$')
ax[0,0].set_title('a)', loc='left')
ax[0,0].minorticks_on()
ax[0,0].set_ylim(-0.2,0.65)
ax[0,0].grid(which='major', axis='y', color='k')
ax[0,0].grid(which='minor', axis='y', linestyle='--')

ax[0,1].scatter(effective_area_ratio, turbine_loss_factor[:,1]/farm_loss_factor[:,1], s=5, c=training_data[:,3])
ax[0,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[0,1].set_ylabel(r'$\Pi_T/\Pi_F$')
ax[0,1].set_title('b)', loc='left')
ax[0,1].minorticks_on()
ax[0,1].set_ylim(-0.2,0.65)
ax[0,1].grid(which='major', axis='y', color='k')
ax[0,1].grid(which='minor', axis='y', linestyle='--')

ax[1,0].scatter(effective_area_ratio, turbine_loss_factor[:,2]/farm_loss_factor[:,2], s=5, c=training_data[:,3])
ax[1,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,0].set_ylabel(r'$\Pi_T/\Pi_F$')
ax[1,0].set_title('c)', loc='left')
ax[1,0].minorticks_on()
ax[1,0].set_ylim(-0.2,0.65)
ax[1,0].grid(which='major', axis='y', color='k')
ax[1,0].grid(which='minor', axis='y', linestyle='--')

ax[1,1].scatter(effective_area_ratio, turbine_loss_factor[:,3]/farm_loss_factor[:,3], s=5, c=training_data[:,3])
ax[1,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[1,1].set_ylabel(r'$\Pi_T/\Pi_F$')
ax[1,1].set_title('d)', loc='left')
ax[1,1].minorticks_on()
ax[1,1].set_ylim(-0.2,0.65)
ax[1,1].grid(which='major', axis='y', color='k')
ax[1,1].grid(which='minor', axis='y', linestyle='--')

ax[2,0].scatter(effective_area_ratio, turbine_loss_factor[:,4]/farm_loss_factor[:,4], s=5, c=training_data[:,3])
ax[2,0].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,0].set_ylabel(r'$\Pi_T/\Pi_F$')
ax[2,0].set_title('e)', loc='left')
ax[2,0].minorticks_on()
ax[2,0].set_ylim(-0.2,0.65)
ax[2,0].grid(which='major', axis='y', color='k')
ax[2,0].grid(which='minor', axis='y', linestyle='--')

ax[2,1].scatter(effective_area_ratio, turbine_loss_factor[:,5]/farm_loss_factor[:,5], s=5, c=training_data[:,3])
ax[2,1].set_xlabel(r'$\lambda/C_{f0}$')
ax[2,1].set_ylabel(r'$\Pi_T/\Pi_F$')
ax[2,1].set_title('f)', loc='left')
ax[2,1].minorticks_on()
ax[2,1].set_ylim(-0.2,0.65)
ax[2,1].grid(which='major', axis='y', color='k')
ax[2,1].grid(which='minor', axis='y', linestyle='--')

plt.tight_layout()
plt.subplots_adjust(wspace=0.5)

cbar = fig.colorbar(pcm, ax=ax.ravel().tolist(), shrink=0.97)
cbar.set_label(r'$\overline{C_T^*}$')

plt.savefig('LES_loss_factor_ratio.png', bbox_inches='tight')

