import numpy as np
import matplotlib.pyplot as plt
import gauss_transformer as tr
import methods as met
import matrix_elements as mat
from sympy import *
from scipy.optimize import minimize
from scipy.linalg import eigh
from svector_class import svector


def energy2Pion(bs,dimG,masses,params,w):
    alphas=[]
    NGP=dimG[0] #Amount of parameters for P-wave Gaussians
    dim=len(w)
    bsP=bs[:NGP] #Parameters for P-wave Gaussians. 
    bsD=bs[NGP:] #Parameters for D-wave Gaussians. Remember that 1 Gaussian = 3 Parameters
    AP=[1/(b**2) for b in bsP]
    for i in range(0,len(bsD),dim):
        A=tr.A_generate(bsD[i:i+dim],w)
        alphas.append(A)
    N, kinetic=mat.PionTwo(AP,alphas,params,masses)
    H=kinetic
    E=eigh(H,N, eigvals_only='true')
    E0=np.amin(E)
    return E0

##Constants and parameters
S_w=41.5 #Coupling strength
b_w=3.9 #Range parameter
m_pi0=134.98 #Neutral pion mass
m_piC=139.570 #Charged pion mass
m_N=1438 #Dressed nucleon mass
hc=197.3 #hbar*c in Mev*fm

mNpi=m_N*m_pi0/(m_N+m_pi0)
mNpipi=m_pi0*(m_N+m_pi0)/(m_N+2*m_pi0)

masses=np.array([m_N,m_pi0,m_piC])
params=np.array([b_w,S_w])
dimP=1
dimD=1
ngausP=1
ngausD=5

E_list=[]
gaussians=[]
E_theoS=[]
bij=np.array([])
bs1=[]
wlist=tr.w_gen_2pion(m_N,m_pi0)
KD=met.K_gen(m_pi0,m_N,m_pi0)

bP=[1]
aP=[1/(b**2) for b in bP]
aD=[np.array([[40,5],[5,10]])]

OM,KM=mat.PionTwo(aP,aD,params,masses,wlist,KD)
OJ,KJ=mat.PionTwo_Test(aP,aD,params,masses,wlist,KD)

print(OM,OJ)

print(np.allclose(OM,OJ))
print(np.allclose(KM,KJ))