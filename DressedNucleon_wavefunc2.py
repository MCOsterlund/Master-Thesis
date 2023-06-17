import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import methods as met
import gauss_transformer as tr
import matrix_elements as mat
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import erfc

##Initial parameters and masses

b=3.9
S=41.5 #Coupling strength parameters
E0=0
mp=938.27 #Mev
mn=939.57
mpi0=134.98 #Mev
mpiC=139.57
mpi=(mpi0+mpiC)/2
mbare=(mp+mn)/2
hc=197.3 #hbar*c
dimP=1
bmax=5

def phi(r,A,c):
    sum=0
    for i in range(1,len(c)):
        sum+=c[i]*np.exp(-A[i-1]*r**2)
    return sum

def energyintegrandP(r,A,c):
    return r**4*S/b*np.exp(-r**2/(b**2))*phi(r,A,c)/c[0]

def normintegrandP(r,A,c):
    return r**4*(phi(r,A,c))**2


def rhoSquared(r,A,c):
    sum=0
    for i in range(len(c)):
        for j in range(len(c)):
            D=A[i-1] + A[j-1]
            D11=D[0,0]
            D12=D[0,1]
            D22=D[1,1]
            term1=np.sqrt(np.pi)*np.exp(D12**2*r**2/D22)*(3*D22**2 + 12*D22*D12**2*r**2+4*D12**4*r**4)*erfc(D12*r/np.sqrt(D22))
            term2=2*np.sqrt(D22)*D12*r*(5*D22+2*D12**2*r**2)
            sum+=c[i]*c[j]*16*np.pi**2/(8*D22**(9/2))*np.exp(-D11*r**2)*r**4*(term1-term2)
    return sum

def rhoNumerical(x,y,A,c):
    sum=0
    for i in range(len(c)):
        for j in range(len(c)):
            D=A[i-1] + A[j-1]
            D11=D[0,0]
            D12=D[0,1]
            D22=D[1,1]
            sum+=c[i]*c[j]*y**4*x**4*np.exp(-D11*x**2 - D22*y**2 - 2*D12*x*y)
    return sum

def intRho(x,A,c):
    return quad(lambda y: rhoNumerical(x,y,A,c), 0, np.inf)[0]

ngausP=2
params=np.array([b,S])
masses=np.array([mbare,mpi])

#Minimizing the one pion system.

dictP=met.global_minP(ngausP,dimP,bmax,masses,params)
masses=dictP['masses']

#Calculte wavefuntion and check the results.

c_end=dictP['eigenvectors'][-1]
coords_end=dictP['coords'][-1]
A_end=[1/(b**2) for b in coords_end]

bmax=5
rmax=5*bmax
rmin=0.01*bmax

start=np.log(rmin)
stop=np.log(rmax)
r=np.logspace(start,stop,num=3000,base=np.exp(1))

resE=quad(energyintegrandP,0,rmax,args=(A_end,c_end,))
resN=quad(normintegrandP,0,rmax,args=(A_end,c_end,))
NP,HP=mat.pion_test_1d(A_end,masses,params)

print('Checking the energy of the wavefunction:', 3*4*np.pi*resE[0], 'Should be:', dictP['E_list'][-1])
print('Checking the norm is unity:', c_end[0]**2 + 3*4*np.pi*resN[0], 'Should be:', c_end.T@NP@c_end)

WavefuncP=phi(r,A_end,c_end)

##Two-pion system.

mpi=(mpi0+mpiC)/2
mbare=(mp+mn)/2 ##Resetting the masses to their starting values.
masses=np.array([mbare,mpi])
ngausD=2
dimP=1
dimG=3
bmax=5
alphaRho=[]


dictD=met.global_minD(ngausP,ngausD,dimP,dimG,bmax,masses,params)

print('Finished D-optimazation, going to numerical integration')



cD=dictD['eigenvectors'][-1]
coordsD=dictD['coords'][-1]

#print('Fuld D coords:', coordsD)
#print('Full eigenvectors:', cD)



alphaPD=[1/(b**2) for b in coordsD[:ngausP]]
cPlotD=cD[:ngausP+1]

cRho=cD[ngausP+1:]
coordsRho=coordsD[ngausP:]

#print('D-wave coords:', coordsD[ngausP:])
masses=dictD['masses']
wlist=tr.w_gen_2pion(masses[0],masses[1])

for j in range(0,len(coordsRho),dimG):
        AD=tr.A_generate(coordsRho[j:j+dimG],wlist)
        alphaRho.append(AD)

#print('Parameters for wavefunction:', alphaD)
#print('Eigenvectors for wavefunction:', cPlotD)

result=np.array([intRho(x,alphaRho,cRho) for x in r])

phiD=phi(r,alphaPD,cPlotD)

resED=quad(energyintegrandP,0,rmax,args=(alphaPD,cPlotD,))
resND=quad(normintegrandP,0,rmax,args=(alphaPD,cPlotD,))
resNRho=quad(intRho,0,rmax,args=(alphaRho,cRho,))
K=met.K_gen(masses[0],masses[1],masses[0])

NDF,HDF=mat.PionTwo(alphaPD,alphaRho,params,masses,wlist,K)

#print('Rho:', rhoSquaredD)

print('Checking the energy of the wavefunction:', 3*4*np.pi*resED[0], 'Should be:', dictD['E_list'][-1])
print('D-wave contribution according to calculated expression:', 9*np.pi**2*resNRho[0], 'should be:', cRho.T@NDF[ngausP+1:,ngausP+1:]@cRho) #Perhaps I should multiply with 9 instead of 3. This gives better results currently though.
print('Checking the norm is unity:', cPlotD[0]**2 + 3*4*np.pi*resND[0]+9*16*np.pi**2*resNRho[0], 'should be:', cD.T@NDF@cD)
print('Checking P-wave contribution:', cPlotD[0]**2 + 3*4*np.pi*resND[0], 'Should be:', cD.T@NDF@cD-cRho.T@NDF[ngausP+1:,ngausP+1:]@cRho)

##Plotting

plt.plot(r,4*np.pi*r**4*np.abs(WavefuncP)**2,color="red",label='One Pion')
plt.plot(r,4*np.pi*r**4*np.abs(phiD)**2,color="blue",label='Two Pions')
plt.plot(r,result, color='green', label='Rho plot')
plt.legend()
plt.show()

    