import numpy as np
from numba import njit
import scipy.integrate as integrate
import scipy.optimize as optimize
import time
import matplotlib.pyplot as plt

teval = np.linspace(0,200,1000)

yiel = 0.1
delta = 0.05
s = 10

initialN = 0.05
initialC = s
initialConditions = [initialN,initialC]

@njit
def chemostat_dynamics(t,y,chemostatArgs):
    mu,k,yiel,delta,s = chemostatArgs
    n,c = y
    dn = mu * n * c / (k + c) - delta * n
    dc = delta*(s-c) - mu * n/yiel * c / (k + c)

    return np.array([dn,dc])

@njit
def glv_rhs(t,y,g,alpha):
    dydot = y*(g - alpha*y)
    return dydot

@njit
def glv_approx_params(c,mu,k,yiel,delta,s):
    sij = mu*k / (k+c)**2
    fi = delta*(s-c) 
    df_de = -delta
    eC = mu*c / (k+c) / yiel
    de_dE = mu*k / (k+c)**2 / yiel

    mij = de_dE / eC * fi - df_de
    gcalc = fi / mij / sij
    alphacalc = eC / sij/mij
    return gcalc,alphacalc

@njit
def logistic_solution(t,g,alpha,x0):
    return x0 * g/alpha  / ((g/alpha - x0)*np.exp(-g*t) + x0)

def glv_error(params,chemostat_population):
    g,alpha = params
    integrated_soln = integrate.solve_ivp(glv_rhs,[0,teval[-1]],[initialN],args=(g,alpha),t_eval=teval,rtol=1e-8,atol=1e-8)
    try:
        error = np.sum((integrated_soln.y[0] - chemostat_population)**2)
    except:
        print(g,alpha)
    return np.sum((integrated_soln.y[0] - chemostat_population)**2)

def integrate_chemostat(teval,initialConditions,chemostatArgs):
    soln = integrate.solve_ivp(chemostat_dynamics,[0,teval[-1]],[initialConditions[0],initialConditions[1]],args=(chemostatArgs,),t_eval=teval)
    return soln

def solve_and_fit(teval,initialConditions,mu,k,yiel,delta,s):
    chemostatArgs = mu,k,yiel,delta,s
    soln = integrate_chemostat(teval,initialConditions,chemostatArgs)

    theory_soln = lambda t,g,alpha: logistic_solution(t,g,alpha,initialConditions[0])
    # full_fit_params,fitcorr  = optimize.curve_fit(theory_soln,teval,soln.y[0],p0=[1,1],bounds=([0,0],[np.inf,np.inf]))


    full_fit_params = optimize.minimize(glv_error,[0.1,0.1],args=(soln.y[0]),method='Nelder-Mead',bounds=((0,None),(0,None))).x
        
    # except:
    #     print(mu,k,yiel,delta,s)
    #     print(mu*initialConditions[1]/(k+initialConditions[1]))
    #     plt.figure()
    #     plt.plot(soln.t,soln.y[0])
    #     plt.show()
    final_qss_approx = glv_approx_params(soln.y[1][-1],*chemostatArgs)

    return full_fit_params,final_qss_approx

muArray = np.round(np.random.uniform(0.1,0.8,20),3)
kArray = np.round(np.random.uniform(1e-1,6,20),3)

fullFitArray = np.zeros((muArray.size,kArray.size,2))
qssFitArray = np.zeros(fullFitArray.shape)

print(time.ctime())
START_TIME = time.time()
for i,mu in enumerate(muArray):
    for j,k in enumerate(kArray):            
            full_fit_params,final_qss_approx = solve_and_fit(teval,initialConditions,mu,k,yiel,delta,s)
            fullFitArray[i,j] = full_fit_params
            qssFitArray[i,j] = final_qss_approx

END_TIME = time.time()
print(time.ctime(),END_TIME-START_TIME)

np.save('../data/fullFitArray_2.npy',fullFitArray)
np.save('../data/qssFitArray_2.npy',qssFitArray)


