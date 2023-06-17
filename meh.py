import numpy as np
import matplotlib.pyplot as plt
import gauss_transformer as tr
import methods as met
import matrix_elements as mat
from sympy import *
from scipy.optimize import minimize
from scipy.linalg import eigh
from scipy.integrate import quad
from svector_class import svector

hbarc=197.327
mp=938.27
mn=939.57
mpi0=134.98
mpiC=139.57 #Constants
E0=0; C0=0 #Initial values of energy and constant (Don't think they are necessary here).
bW=3.9; SW=41.5

def phi(r,A,c): #Function for phi
    sum=0
    for i in range(1,len(c)):
        sum+=c[i]*np.exp(-A[i-1]*r**2)
    return sum

def E_pionphoto(A,masses,params): #Energy function
    N,H=pion_test_1d(A,masses,params)
    E,c=eigh(H,N, subset_by_index=[0,0])
    c0=c
    return E,c0

def minfunc(A,masses,params):
    E,c=E_pionphoto(A,masses,params)
    return E
    
def pion_test_1d(alphas,masses,params): #Creation of matrix elements (THESE ARE CORRECT!)
    b_w=params[0]
    S_w=params[1]
    m_N=masses[0]
    m_pi=masses[1]
    m_Npi=m_N*m_pi/(m_N+m_pi)
    kap=1/b_w**2
    kap=np.identity(1)*kap
    length=len(alphas)+1 #Make dimension one greater due to hardcoding parameter
    kinetic=np.zeros((length,length))
    overlap=np.zeros((length,length)) #Initialize matrices
    overlap[0,0]=1
    kinetic[0,0]=0 #Hardcoding parameters
    for i in range(length):
        for j in range(length):
            if j<=i:
                if i==0 and j==0:
                    continue
                elif j==0 and i!=0: ##Creation elements
                    B=alphas[i-1]
                    overlap[i,j]=0
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=3*S_w/b_w*1.5/(B+kap)*(np.pi/(B+kap))**(3/2)
                    kinetic[j,i]=kinetic[i,j]
                else: ##Kinetic terms
                    A=alphas[i-1]; B=alphas[j-1]
                    overlap[i,j]=3*1.5/(B+A)*(np.pi/(B+A))**(3/2)
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=3*(197.3)**2/(2*m_Npi)*15*A*B/((A+B)**2)*(np.pi/(A+B))**(3/2)+m_pi*overlap[i,j]
                    kinetic[j,i]=kinetic[i,j]
    return overlap, kinetic

def optimize(As,masses,params):
    #I think optimization oversteps the mark. 
    resS=minimize(minfunc, As, args=(masses,params,), method="Nelder-Mead",options={'adaptive': true}) #Optimize parameters
    E0S,c=E_pionphoto(resS.x,masses,params)
    A=resS.x
    return E0S,c,A

##-------------------ANALYTIC INTEGRAL CHECKS--------------------------------

def energyintegrand(r,A,c):
    return r**4*SW/bW*np.exp(-r**2/(bW**2))*phi(r,A,c)/c[0]

def normintegrand(r,A,c):
    return r**4*(phi(r,A,c))**2

bmax=5
ngaus=4
dim=1
params=np.array([bW,SW])
rmax=5*bmax
rmin=0.01*bmax

start=np.log(rmin)
stop=np.log(rmax)
r=np.logspace(start,stop,num=3000,base=np.exp(1))
As2=AC=np.array([0.095,0.064,0.056,0.079])
cC=np.array([-0.999932,4.1e-06, 0.012, 0.0014, 3.8e-05]).reshape((5,1))
#print('Initial A:', As)
bs1=[]

plt.figure(0)
for i in range(ngaus):
    hal=tr.halton(i+1,dim)
    bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.
    bs1=np.append(bs1,bs)
    As=[1/(b**2) for b in bs1]
    masses=[(mp+mn)/2-E0,(mpi0+mpiC)/2]
    #Add generated element to basis
    E0,Co,A=optimize(As,masses,params)
    plt.plot(r,phi(r,A,Co)/Co[0])

print('CHECKING INTEGRALS OF THE WAVE FUNCTION.\n')

Nc,Hc=pion_test_1d(A,masses,params)
resE=quad(energyintegrand,0,rmax,args=(A,Co,))
resN=quad(normintegrand,0,rmax,args=(A,Co,))

resEtot=3*4*np.pi*resE[0]
resNtot=3*4*np.pi*resN[0]

print('Energy from integral is:', resEtot, '. Should be:',E0)
print('Integrated norm is:', Co[0]**2+resNtot, '. Should be roughly:', Co.T@Nc@Co)

#print('Eo:', Eo, 'Co:', Co)
plt.plot(r,-phi(r,As2,cC))
plt.show()
