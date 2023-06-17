##KEEP IT SIMPLE!

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, det
from scipy.optimize import minimize
from scipy.integrate import quad
import matrix_elements as mat
from svector_class import svector
import gauss_transformer as tr

##Exact wavefunctions

def GaussSquared(r,alphas,c):
    sum=0
    for i in range(len(alphas)):
        sum+=c[i]*np.exp(-alphas[i]*r**2)
    return r**2*sum**2

def GaussSquaredP(r,alphas,c):
    sum=0
    alphas=mat.transform_list(alphas)
    for i in range(len(alphas)):
        for j in range(len(alphas)):
            A=alphas[i]; B=alphas[j]
            sum+=1/np.sqrt(A)*1/np.sqrt(B)*4/3*np.pi*r**2*c[i]*c[j]*np.exp(-(A+B)*r**2)
    return r**2*sum

def GaussSquaredD(r,alphas,c):
    sum=0
    alphas=mat.transform_list(alphas)
    for i in range(len(alphas)):
        for j in range(len(alphas)):
            A=alphas[i]; B=alphas[j]
            sum+=1/A*1/B*16/15*np.pi/4*r**4*c[i]*c[j]*np.exp(-(A+B)*r**2)
    return r**2*sum

def SExact(r):
    return 2*np.exp(-r)

def PExact(r):
    return 1/(np.sqrt(6)*2)*(r)*np.exp(-r/2)

def DExact(r):
    return 4/(np.sqrt(30)*81)*(r**2)*np.exp(-r/3)

def SSquared(r):
    return r**2*SExact(r)**2

def PSquared(r):
    return r**2*PExact(r)**2

def DSquared(r):
    return r**2*DExact(r)**2

intS=quad(SSquared,0,np.inf)
intP=quad(PSquared,0,np.inf)
intD=quad(DSquared,0,np.inf)

print('Integrating S-waves:', intS[0], '\n')
print('Integrating P-waves:', intP[0], '\n')
print('Integrating D-waves:', intD[0], '\n')

##Gaussians and test functions for energy calculation

def S_test(blist):
    alphas=[1/(b**2) for b in blist]
    length=len(alphas)
    kinetic=np.zeros((length,length))
    overlap=np.zeros((length,length))
    coulomb=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            if j<=i:
                A=alphas[i]; B=alphas[j]
                D=A+B
                overlap[i,j]=(np.pi/D)**(3/2)
                overlap[j,i]=overlap[i,j]
                kinetic[i,j]=-3*(np.pi**(3.0/2))*A*(A-D)/(D**(5.0/2))
                kinetic[j,i]=kinetic[i,j]
                coulomb[i,j]=-2*np.pi/D
                coulomb[j,i]=coulomb[i,j]
            else:
                break
    H=kinetic+coulomb
    E=eigh(H,overlap,eigvals_only='true')
    E0=np.amin(E)
    return E0

def P_test(blist):
    alphas=[1/(b**2) for b in blist]
    length=len(alphas)
    kinetic=np.zeros((length,length))
    overlap=np.zeros((length,length))
    coulomb=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            if j<=i:
                A=alphas[i]; B=alphas[j]
                D=A+B
                overlap[i,j]=((np.pi)**(3.0/2))/(2*D**(5.0/2))
                overlap[j,i]=overlap[i,j]
                kinetic[i,j]=-5/2*(np.pi**(3.0/2))*A*(A-D)/(D**(7.0/2))
                kinetic[j,i]=kinetic[i,j]
                coulomb[i,j]=-2*np.pi/(3*D**2)
                coulomb[j,i]=coulomb[i,j]
            else:
                break
    H=kinetic+coulomb
    E=eigh(H,overlap,eigvals_only='true')
    E0=np.amin(E)
    return E0

def D_test(blist):
    alphas=[1/(b**2) for b in blist]
    length=len(alphas)
    kinetic=np.zeros((length,length))
    overlap=np.zeros((length,length))
    coulomb=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            if j<=i:
                A=alphas[i]; B=alphas[j]
                D=A+B
                overlap[i,j]=((np.pi)**(3.0/2))/(4*D**(7.0/2))
                overlap[j,i]=overlap[i,j]
                kinetic[i,j]=-7/4*(np.pi**(3.0/2))*A*(A-D)/(D**(9.0/2))
                kinetic[j,i]=kinetic[i,j]
                coulomb[i,j]=-4*np.pi/(15*D**3)
                coulomb[j,i]=coulomb[i,j]
            else:
                break
    H=kinetic+coulomb
    E=eigh(H,overlap,eigvals_only='true')
    E0=np.amin(E)
    return E0

def energyS(blist, K):
    alphas=[1/(b**2) for b in blist]
    w=np.ones((1,1))
    N, kin, Coul=mat.S_wave(alphas,K,w)
    H=kin+Coul
    E,c=eigh(H,N,subset_by_index=[0,0])
    return E,c

def minfuncS(blist,K):
    E,c=energyS(blist,K)
    return E

def minfuncP(blist,K):
    E,c=energyP(blist,K)
    return E

def minfuncD(blist,K):
    E,c=energyD(blist,K)
    return E

def energyP(blist,K):
    alphas=[1/(b**2) for b in blist]
    w=np.ones((1,1))
    N, kin, Coul=mat.P_wave(alphas,K,w)
    H=kin-Coul
    E,c=eigh(H,N,subset_by_index=[0,0])
    return E,c

def energyD(blist,K):
    alphas=[1/(b**2) for b in blist]
    w=np.ones((1,1))
    N, kin, Coul=mat.D_wave(alphas,K,w)
    H=kin-Coul
    E,c=eigh(H,N,subset_by_index=[0,0])
    return E,c

n=1
K=np.identity(n)*0.5
w=np.ones((1,1))
bmax1=5
bmax2=9
bmax3=12
ngaus=5
gaussians=[]
E_minS=[]
E_minS_test=[]
E_minP=[]
E_minP_test=[]
E_minD=[]
E_minD_test=[]
E_theoS=[]
E_theoP=[]
E_theoD=[]
wavefunc=[]
S1=[]
P1=[]
D1=[]
cS1=[]
cP1=[]
cD1=[]


for i in range(ngaus):
    bs1=[bmax1*(j+1)/(i+1) for j in range(i+1)]
    bs2=[bmax2*(j+1)/(i+1) for j in range(i+1)]
    bs3=[bmax3*(j+1)/(i+1) for j in range(i+1)]
    resS=minimize(minfuncS, bs1, args=(K,), options={'maxiter': 1000})
    #resS_test=minimize(S_test,bs1, options={'maxiter': 1000})
    resP=minimize(minfuncP, bs2, args=(K,), options={'maxiter': 1000})
    #resP_test=minimize(P_test, bs2, options={'maxiter': 1000})
    resD=minimize(minfuncD, bs3, args=(K,), options={'maxiter': 1000})
    #resD_test=minimize(D_test, bs3, options={'maxiter': 1000})
    E0S,cS=energyS(resS.x,K)
    #E0S_test=S_test(resS_test.x)
    E0P,cP=energyP(resP.x,K)
    #E0P_test=P_test(resP_test.x)
    E0D,cD=energyD(resD.x,K)
    #E0D_test=D_test(resD_test.x)
    E_minS=np.append(E_minS,E0S)
    #E_minS_test=np.append(E_minS_test,E0S_test)
    E_minP=np.append(E_minP,E0P)
    #E_minP_test=np.append(E_minP_test,E0P_test)
    E_minD=np.append(E_minD,E0D)
    #E_minD_test=np.append(E_minD_test,E0D_test)
    S1.append(resS.x)
    P1.append(resP.x)
    D1.append(resD.x)
    cS1.append(cS)
    cP1.append(cP)
    cD1.append(cD)
    E_theoS.append(-0.5)
    E_theoP.append(-0.125)
    E_theoD.append(-0.055555)
    gaussians.append(i+1)

print('Final ratio of S-wave energy compared to actual value:', (1-np.abs(E0S)/(0.5))*100, '\n')

print('Final ratio of P-wave energy compared to actual value:', (1-np.abs(E0P)/(0.125))*100, '\n')

print('Final ratio of D-wave energy compared to actual value:', (1-np.abs(E0D)/(0.055555))*100, '\n')

alphasS=[1/(b**2) for b in resS.x]
alphasP=[1/(b**2) for b in resP.x]
alphasD=[1/(b**2) for b in resD.x]

aS1=[]
aP1=[]
aD1=[]
for i in range(len(S1)):
    A1=np.array([1/(b**2) for b in S1[i]])
    B1=np.array([1/(b**2) for b in P1[i]])
    C1=np.array([1/(b**2) for b in D1[i]])
    aS1.append(A1)
    aP1.append(B1)
    aD1.append(C1)

NP,HP,coulP=mat.P_wave(alphasP,K,w)

intGaussS=quad(GaussSquared,0,np.inf,args=(alphasS,cS,))
intGaussP=quad(GaussSquaredP,0,np.inf,args=(alphasP,cP,))
intGaussD=quad(GaussSquaredD,0,np.inf,args=(alphasD,cD,))

print('Integral of S-Gaussians:', 4*np.pi*intGaussS[0], '\n')
print('Integral of P-Gaussians:', intGaussP[0], 'Matrix method:', cP.T@NP@cP, '\n')
print('Integral of D-Gaussians:', intGaussD[0])

##TEXT SIZE CUSTOMIZATION
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rcParams['axes.titlesize']=BIGGER_SIZE
plt.rcParams['legend.fontsize']=MEDIUM_SIZE
plt.rcParams['axes.labelsize']=MEDIUM_SIZE
plt.rcParams['xtick.labelsize']=SMALL_SIZE
plt.rcParams['ytick.labelsize']=SMALL_SIZE

plt.figure(1)
plt.plot(gaussians,E_minS, marker='o')
#plt.plot(gaussians,E_minS_test, marker='v')
plt.plot(gaussians,E_theoS, '--')
plt.title('S-wave convergence of Hydrogen')
plt.xlabel('Number of Gaussians')
plt.ylabel('Energy [Hartree]')
plt.legend(['Numerical result', 'Theoretical value'])
plt.savefig('figures/swave.pdf'.format(1),bbox_inches='tight')


plt.figure(2)
plt.plot(gaussians,E_minP, marker='o')
#plt.plot(gaussians,E_minP_test, marker='v')
plt.plot(gaussians,E_theoP, '--')
plt.title('P-wave convergence of Hydrogen')
plt.yticks([-0.125, -0.123, -0.121, -0.119, -0.117, -0.115, -0.113])
plt.xlabel('Number of Gaussians')
plt.ylabel('Energy [Hartree]')
plt.legend(['Numerical result', 'Theoretical value'])
plt.savefig('figures/pwave.pdf'.format(2),bbox_inches='tight')


plt.figure(3)
plt.plot(gaussians,E_minD, marker='o')
#plt.plot(gaussians,E_minD_test, marker='v')
plt.plot(gaussians,E_theoD, '--')
plt.title('D-wave convergence of Hydrogen')
plt.xlabel('Number of Gaussians')
plt.ylabel('Energy [Hartree]')
plt.legend(['Numerical result', 'Theoretical value'])
plt.savefig('figures/dwave.pdf'.format(3),bbox_inches='tight')


##PLOTTING WAVEFUNCTIONS

rmin=0.01
rmaxS=10
rmaxP=17
rmaxD=23
start=np.log(rmin)
stopS=np.log(rmaxS)
stopP=np.log(rmaxP)
stopD=np.log(rmaxD)
rS=np.logspace(start,stopS,num=3000,base=np.exp(1))
rP=np.logspace(start,stopP,num=3000,base=np.exp(1))
rD=np.logspace(start,stopD,num=3000,base=np.exp(1))

GaussP=np.reshape(GaussSquaredP(rP,alphasP,cP),(3000,))
GaussD=np.reshape(GaussSquaredD(rD,alphasD,cD),(3000,))

plt.figure(4)   
plt.title('Radial wavefunctions')
plt.ylabel(r'$rR_{nl}(r)$')
plt.xlabel(r'r [a]')
plt.plot(rS,SExact(rS),label='S-wave: n=1, l=0')
plt.plot(rS,PExact(rS),label='P-wave: n=2, l=0')
plt.plot(rS,DExact(rS),label='D-wave: n=3, l=0')
plt.legend()
plt.savefig('figures/HydrogenWavefuncs.pdf'.format(4),bbox_inches='tight')


plt.figure(5)
plt.title('S-wave Wavefunction')
plt.ylabel(r'$|R_{10}r|^2$')
plt.xlabel('r [a]')
plt.plot(rS,SSquared(rS),label='Exact solution')
plt.plot(rS,np.reshape(4*np.pi*GaussSquared(rS,aS1[0],cS1[0]),(3000,)), '--', label='1 Gaussian')
plt.plot(rS,np.reshape(4*np.pi*GaussSquared(rS,aS1[2],cS1[2]),(3000,)), '--', label='3 Gaussians')
plt.plot(rS,4*np.pi*np.abs(GaussSquared(rS,alphasS,cS)), '--', label='5 Gaussians')
plt.legend()
plt.savefig('figures/HydrogenSwaveComp.pdf'.format(5),bbox_inches='tight')

plt.figure(6)
plt.title('P-wave wavefunction')
plt.ylabel(r'$|R_{21}r|^2$')
plt.xlabel('r [a]')
plt.plot(rP,PSquared(rP),label='Exact solution')
plt.plot(rP,np.reshape(GaussSquaredP(rP,aP1[0],cP1[0]),(3000,)), '--', label='1 Gaussian')
plt.plot(rP,np.reshape(GaussSquaredP(rP,aP1[2],cP1[2]),(3000,)), '--', label='3 Gaussians')
plt.plot(rP,GaussP, '--', label='5 Gaussians')
plt.legend()
plt.savefig('figures/HydrogenPwaveComp.pdf'.format(6),bbox_inches='tight')

plt.figure(7)
plt.title('D-wave wavefunction')
plt.ylabel(r'$|R_{32}r|^2$')
plt.xlabel('r [a]')
plt.plot(rD,DSquared(rD),label='Exact solution')
plt.plot(rP,np.reshape(GaussSquaredD(rD,aD1[0],cD1[0]),(3000,)), '--', label='1 Gaussian')
plt.plot(rP,np.reshape(GaussSquaredD(rD,aD1[2],cD1[2]),(3000,)), '--', label='3 Gaussians')
plt.plot(rD,GaussD, '--', label='5 Gaussians')
plt.legend()
plt.savefig('figures/HydrogenDwaveComp.pdf'.format(7),bbox_inches='tight')
plt.show()