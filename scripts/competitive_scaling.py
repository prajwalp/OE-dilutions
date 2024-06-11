import numpy as np
import scipy.special as special
import scipy.integrate as integrate
import scipy.optimize as optimize

import sys
sys.path.append("../modules/")
import competition_functions as cmpfns

def gamma_ccdf(x,a,b):
    return special.gammaincc(a,x/b)

def gamma_fn(x,k,theta):
    return x**(k-1) * np.exp(-x/theta) / (special.gamma(k) * theta**k)


def generate_mu(Ns,Nr,mean,sd):
    muMatrix =  np.random.lognormal(mean,sd,(Ns,Nr))
    muMatrix =  np.round(muMatrix / np.max(muMatrix) *0.8,3)
    return muMatrix

def generate_k(Ns,Nr,mean,sd):
    kMatrix = np.round(np.random.lognormal(mean,sd,(Ns,Nr))/10,3)
    return kMatrix

def mainGammaFitFn(Nr,Ns,supplyVec,delta,muMean,muSd,kMean,kSd):
    muMatrix = generate_mu(Ns,Nr,muMean,muSd)
    kMatrix = generate_k(Ns,Nr,kMean,kSd)

    initialPopulations = np.full(Ns,0.05)
    initialResources = supplyVec
    initialConditions = np.concatenate((initialPopulations,initialResources))

    t = np.linspace(0,1000,1000)
    
    chemostat_sol = integrate.solve_ivp(cmpfns.chemostat_dynamics,(0,t[-1]),initialConditions,args=(muMatrix, kMatrix, delta, supplyVec,Nr,Ns),t_eval=t,max_step=0.1)

    qssCEnd = cmpfns.qssResourcesSolver(cmpfns.resUsage,chemostat_sol.y[:Ns,-1],chemostat_sol.y[Ns:,-1],muMatrix, kMatrix, delta, supplyVec,Nr,Ns)
    tdepGrowthEnd,tdepInterEnd = cmpfns.glvParamsFn(muMatrix,kMatrix,qssCEnd,delta,supplyVec,Nr,Ns)

    glvSoln = cmpfns.solveGLV(initialPopulations,t,tdepGrowthEnd,tdepInterEnd)
    glvChemostatDiff = (glvSoln[:,-1] - chemostat_sol.y[:Ns,-1])

    histend,edgesend = np.histogram(tdepInterEnd.flatten(),bins="auto",density=False)
    cumhistend = 1-np.cumsum(histend)/np.sum(histend)
    fitend = optimize.curve_fit(gamma_ccdf,edgesend[:-1],cumhistend)

    return muMatrix,kMatrix,tdepGrowthEnd,tdepInterEnd,fitend,chemostat_sol.y[:,-1],glvChemostatDiff

def multipleRun(Nr,Ns,supplyVec,delta,muMean,muSd,kMean,kSd,numRuns):
    fitResults = np.zeros((numRuns,2))
    fitErrors = np.zeros((numRuns,2,2))
    muMatrices = np.zeros((numRuns,Ns,Nr))
    kMatrices = np.zeros((numRuns,Ns,Nr))
    tdepGrowths = np.zeros((numRuns,Ns))
    tdepInters = np.zeros((numRuns,Ns,Ns))
    chemostatSolutions = np.zeros((numRuns,Ns+Nr))
    glvErrors = np.zeros((numRuns,Ns))
    for i in range(numRuns):
        muMatrices[i],kMatrices[i],tdepGrowths[i],tdepInters[i],fitend,chemostatSolutions[i],glvErrors[i]= mainGammaFitFn(Nr,Ns,supplyVec,delta,muMean,muSd,kMean,kSd)
        fitResults[i],fitErrors[i] = fitend[0],fitend[1]
    return muMatrices,kMatrices,tdepGrowths,tdepInters,fitResults,fitErrors,chemostatSolutions,glvErrors


