import numpy as np
from numba import njit,prange

def assertParams(muMatrix,kMatrix,dTensor,delta,supplyVec,Nr,Ns):
    assert np.shape(muMatrix) == (Ns,Nr)
    assert np.shape(kMatrix) == (Ns,Nr)
    assert np.shape(dTensor) == (Ns,Nr,Nr)
    # assert np.shape(rVec) == (Nr,)
    assert np.shape(supplyVec) == (Nr,)

    assert np.all(supplyVec >= 0)
    assert delta > 0
    assert np.all(muMatrix >= 0)
    assert np.all(kMatrix > 0)
    assert np.all(dTensor >= 0)

@njit(parallel=True)
def chemostat_dynamics(t, y, muMatrix, kMatrix, dTensor, delta, supplyVec,Nr,Ns):
    populations = y[:Ns]
    resources = y[Ns:]

    dpop = np.zeros(Ns,dtype=np.float64)
    dres = np.zeros(Nr,dtype=np.float64)
    uptakeMatrix = (muMatrix.T * populations).T * resources / (kMatrix + resources)

    dpop = (np.sum(uptakeMatrix,axis=1) - delta*populations)    
    resourceUsage = np.sum(uptakeMatrix,axis=0)  
    
    for alpha in prange(Nr):        
        leakage = np.sum(dTensor[:,alpha,:]*uptakeMatrix)            
        dres[alpha] = delta*(supplyVec[alpha] - resources[alpha]) - resourceUsage[alpha] + leakage

    return np.concatenate((dpop,dres))

def gvecFn(muMatrix,kMatrix,rVec,Nr,Ns):
    gvec = np.zeros(Ns)
    for i in range(Ns):
        for alpha in range(Nr):
            gvec[i] += muMatrix[i,alpha]*rVec[alpha] / (kMatrix[i,alpha] + rVec[alpha])
    return gvec

def fVectorFn(supplyVec,rVec,dilutionRate,Nr,Ns):
    return dilutionRate*(supplyVec - rVec)

def eMatrixFn(muMatrix,kMatrix,dTensor,rVec,Nr,Ns):
    eMatrix = np.zeros((Nr,Ns))
    for beta in range(Nr):
        for j in range(Ns):
            leakage = 0
            for alpha in range(Nr):
                leakage += dTensor[j,beta,alpha]*muMatrix[j,alpha]*rVec[alpha] / (kMatrix[j,alpha] + rVec[alpha])
            eMatrix[beta,j] = leakage + muMatrix[j,beta] * rVec[beta] / (kMatrix[j,beta] + rVec[beta])
    return eMatrix

def sMatrixFn(muMatrix,kMatrix,rVec,Nr,Ns):
    sMatrix = np.zeros((Nr,Ns))
    for beta in range(Nr):
        for j in range(Ns):
            sMatrix[beta,j] = muMatrix[j,beta] * kMatrix[j,beta] / (kMatrix[j,beta] + rVec[beta])**2
    return sMatrix

def dedrFn(muMatrix,kMatrix,dTensor,rVec,Nr,Ns):
    dedr = np.zeros((Nr,Nr,Ns))
    for beta in range(Nr):
        for alpha in range(Nr):
            for j in range(Ns):
                dedr[beta,alpha,j] = dTensor[j,beta,alpha]*muMatrix[j,alpha]*kMatrix[j,alpha] / (kMatrix[j,alpha] + rVec[alpha])**2
            if beta == alpha:
                dedr[beta,alpha,j] += muMatrix[j,alpha] *kMatrix[j,alpha]/ (kMatrix[j,alpha] + rVec[alpha])**2
    return dedr

def mMatrixFn(muMatrix,kMatrix,dTensor,rVec,delta,supplyVec,Nr,Ns):
    mMatrix = np.zeros((Nr,Nr))
    eMatrix = eMatrixFn(muMatrix,kMatrix,dTensor,rVec,Nr,Ns)
    fVec = fVectorFn(supplyVec,rVec,delta,Nr,Ns)
    dedr = dedrFn(muMatrix,kMatrix,dTensor,rVec,Nr,Ns)

    eMatrixInv = np.linalg.pinv(eMatrix)

    for alpha in range(Nr):
        for beta in range(Nr):
            doublesum = 0
            for i in range(Ns):
                for gamma in range(Nr):
                    doublesum += dedr[alpha,beta,i]*eMatrixInv[i,gamma]*fVec[gamma]
            mMatrix[alpha,beta] = doublesum - delta
    return mMatrix

def glvParamsFn(muMatrix,kMatrix,dTensor,rVec,delta,supplyVec,Nr,Ns):
    fVec = fVectorFn(supplyVec,rVec,delta,Nr,Ns)
    eMatrix = eMatrixFn(muMatrix,kMatrix,dTensor,rVec,Nr,Ns)
    sMatrix = sMatrixFn(muMatrix,kMatrix,rVec,Nr,Ns)
    mMatrix = mMatrixFn(muMatrix,kMatrix,dTensor,rVec,delta,supplyVec,Nr,Ns)
    mInv = np.linalg.inv(mMatrix)

    growthVec = np.dot(sMatrix.T,np.dot(mInv,fVec))
    interactionMatrix = np.dot(-sMatrix.T,np.dot(mInv,eMatrix))
    return growthVec,interactionMatrix