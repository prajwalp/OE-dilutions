import numpy as np
from numba import njit,prange
import scipy.optimize as optimize
import scipy.integrate as integrate

def assertParams(muMatrix,kMatrix,delta,supplyVec,Nr,Ns):
    assert np.shape(muMatrix) == (Ns,Nr)
    assert np.shape(kMatrix) == (Ns,Nr)
    # assert np.shape(rVec) == (Nr,)
    assert np.shape(supplyVec) == (Nr,)

    assert np.all(supplyVec >= 0)
    assert delta > 0
    assert np.all(muMatrix >= 0)
    assert np.all(kMatrix > 0)

@njit(parallel=True)
def chemostat_dynamics(t, y, muMatrix, kMatrix, delta, supplyVec,Nr,Ns):
    populations = y[:Ns]
    resources = y[Ns:]

    dpop = np.zeros(Ns,dtype=np.float64)
    dres = np.zeros(Nr,dtype=np.float64)
    uptakeMatrix = (muMatrix.T * populations).T * resources / (kMatrix + resources)

    dpop = (np.sum(uptakeMatrix,axis=1) - delta*populations)      
    dres = delta*(supplyVec - resources) - np.sum(uptakeMatrix,axis=0)    

    return np.concatenate((dpop,dres))

def gvecFn(muMatrix,kMatrix,rVec,delta,Nr,Ns):
    gvec = np.zeros(Ns)
    for i in range(Ns):
        for alpha in range(Nr):
            gvec[i] += muMatrix[i,alpha]*rVec[alpha] / (kMatrix[i,alpha] + rVec[alpha])
    return gvec - delta

def sigmaVectorFn(supplyVec,rVec,dilutionRate,Nr,Ns):
    return dilutionRate*(supplyVec - rVec)

def fMatrixfn(muMatrix,kMatrix,rVec,Nr,Ns):
    fMatrix = np.zeros((Nr,Ns))
    for beta in range(Nr):
        for j in range(Ns):
            fMatrix[beta,j] =  - muMatrix[j,beta] * rVec[beta] / (kMatrix[j,beta] + rVec[beta])
    return fMatrix

def sMatrixFn(muMatrix,kMatrix,rVec,Nr,Ns):
    sMatrix = np.zeros((Nr,Ns))
    for beta in range(Nr):
        for j in range(Ns):
            sMatrix[beta,j] = muMatrix[j,beta] * kMatrix[j,beta] / (kMatrix[j,beta] + rVec[beta])**2
    return sMatrix

def dfdrFn(muMatrix,kMatrix,rVec,Nr,Ns):
    dfdr = np.zeros((Nr,Nr,Ns))
    for beta in range(Nr):
            for j in range(Ns):
                dfdr[beta,beta,j] += -muMatrix[j,beta] *kMatrix[j,beta]/ (kMatrix[j,beta] + rVec[beta])**2
    return dfdr

def mMatrixFn(muMatrix,kMatrix,rVec,delta,supplyVec,Nr,Ns):
    mMatrix = np.zeros((Nr,Nr))
    fMatrix = fMatrixfn(muMatrix,kMatrix,rVec,Nr,Ns)
    sigmaVec = sigmaVectorFn(supplyVec,rVec,delta,Nr,Ns)
    dfdr = dfdrFn(muMatrix,kMatrix,rVec,Nr,Ns)

    fMatrixInv = np.linalg.pinv(fMatrix)

    for alpha in range(Nr):
        for beta in range(Nr):
            doublesum = 0
            for i in range(Ns):
                for gamma in range(Nr):
                    doublesum += dfdr[alpha,beta,i]*fMatrixInv[i,gamma]*sigmaVec[gamma]
            mMatrix[alpha,beta] = doublesum + delta*(alpha == beta)
    return mMatrix

def glvParamsFn(muMatrix,kMatrix,rVec,delta,supplyVec,Nr,Ns):
    sigmaVec = sigmaVectorFn(supplyVec,rVec,delta,Nr,Ns)
    fMatrix = fMatrixfn(muMatrix,kMatrix,rVec,Nr,Ns)
    sMatrix = sMatrixFn(muMatrix,kMatrix,rVec,Nr,Ns)
    mMatrix = mMatrixFn(muMatrix,kMatrix,rVec,delta,supplyVec,Nr,Ns)
    mInv = np.linalg.inv(mMatrix)

    growthVec = np.dot(sMatrix.T,np.dot(mInv,sigmaVec)) + gvecFn(muMatrix,kMatrix,rVec,delta,Nr,Ns)
    interactionMatrix = np.dot(-sMatrix.T,np.dot(mInv,fMatrix))
    return growthVec,interactionMatrix

def resUsage(resources,populations,muMatrix, kMatrix, delta, supplyVec,Nr,Ns):
    dres = np.zeros(Nr)
    uptakeMatrix = (muMatrix.T * populations).T * resources / (kMatrix + resources)
    resourceUsage = np.sum(uptakeMatrix,axis=0)  
    
    dres = delta*(supplyVec - resources) - resourceUsage

    return dres

def qssResourcesSolver(resourceUsageFn,populations,guessResource,muMatrix, kMatrix, delta, supplyVec,Nr,Ns):
    qssSolution = optimize.least_squares(resourceUsageFn,guessResource,args=(populations,muMatrix, kMatrix, delta, supplyVec,Nr,Ns),bounds=(0,np.inf))

    if(qssSolution.success):
        return qssSolution.x
    else:
        print("Failed to find QSS solution")
        return supplyVec
    

@njit
def timeDepGLV(t,y,teval,gTime,alphaTime):
    indexT = np.where(teval >= t)[0][0]
    g,alpha = gTime[indexT],alphaTime[indexT]
    return y*(g - np.dot(alpha,y))

@njit
def glvFn(t,y,teval,g,alpha):
    return y*(g - np.dot(alpha,y))

def solveTimeDepGLV(y0,teval,gTime,alphaTime):
    sol = integrate.solve_ivp(timeDepGLV,(teval[0],teval[-1]),y0,args=(teval,gTime,alphaTime),t_eval=teval,method='RK45',rtol=1e-8,atol=1e-8)
    if(not sol.success):
        print("Failed to solve time dependent GLV")
    return sol.y

def solveGLV(y0,teval,g,alpha):
    sol = integrate.solve_ivp(glvFn,(teval[0],teval[-1]),y0,args=(teval,g,alpha),t_eval=teval,method='RK45',rtol=1e-8,atol=1e-8)
    if(not sol.success):
        print("Failed to solve time dependent GLV")
    return sol.y