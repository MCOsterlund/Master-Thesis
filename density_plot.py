import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import methods as met
from svector_class import svector
import gauss_transformer as tr
import matrix_elements as mat
from scipy.optimize import minimize
from scipy.integrate import quad

def transform_list(alphas):
    g_new=[np.ones((1,1))*alphas[i] for i in range(len(alphas))]
    return g_new

def D_elem_delta(r,a,b,c,d,A,B,wlist):
    res=0
    dim=A.shape[0]
    D=A+B
    R=np.linalg.inv(D)
    P=(R@wlist[0])@(wlist[0].T@R)
    beta=1/(wlist[0].T@R@wlist[0])
    M0=((np.pi**dim)/(np.linalg.det(D)))**(3/2)
    J0=(beta/np.pi)**(3/2)*np.exp(-beta*r**2)
    M2=1/4*((a.T@R@b)*(c.T@R@d) + (a.T@R@c)*(b.T@R@d) + (a.T@R@d)*(b.T@R@c))*M0
    J2=4/15*beta**2*((a.T@P@b)*(c.T@P@d) + (a.T@P@c)*(b.T@P@d) + (a.T@P@d)*(b.T@P@c))*(4*beta**2*r**4 - 20*beta*r**2 + 15)*J0
    M1J1=2/3*beta*(2*beta*r**2 -3)*((a.T@P@b)*(c.T@R@d) + (a.T@P@c)*(b.T@R@d) + (a.T@P@d)*(b.T@R@c) + (b.T@P@c)*(a.T@R@d) + (b.T@P@d)*(a.T@R@c) + (c.T@P@d)*(a.T@R@b))*M0*J0
    res+=M2*J0 +M1J1+M0*J2
    return res

##Change D-waves above and try what happens with the density when I insert CoM coordinate instead.

def P_elem_delta(r,a,b,A,B):
    dim=A.shape[0]
    D=A+B
    R=np.linalg.inv(D)
    P=R@R
    beta=1/(R@R)
    M0=(np.pi**dim/(np.linalg.det(D)))**(3/2)
    M1=1/2*(a*R*b)*M0
    J0=4*(beta/np.pi)**(3/2)*np.pi*r**2*np.exp(-beta*r**2)
    J1=1/3*beta*(a*P*b)*(2*beta*r**2 - 3)*J0
    res=M1*J0 + M0*J1
    return res

#def DensityFunc(x,y,c,alphaP,alphaD,wlist):
#    lengthP=len(alphaP)
#    lengthD=len(alphaD)
#    alphaP=transform_list(alphaP)
#    dimP=alphaP[0].shape[0]
#    ap,bp,bHp=svector.create_pion_shift(dimP)
#    res=0
#    for i in range(lengthP+lengthD):
#        for j in range(lengthP+lengthD):
#            if i<lengthP and j<lengthP:
#                Api=alphaP[i]; Apj=alphaP[j]
#                P1=P_elem_delta(x,ap,ap,Api,Apj)
#                P2=P_elem_delta(x,bp,bHp,Api,Apj)
#                res+=c[i+1]*c[j+1]*(P1+P2)
#            elif i<lengthP and j>=lengthP:
#                continue
#            elif j<lengthP and i>=lengthP:
#                continue
#            else:
#                Adi=alphaD[i-lengthP]; Adj=alphaD[j-lengthP]
#                D1=D_elem_delta(x,y,ap,ap,ap,ap,Adi,Adj,wlist)
#                D2=D_elem_delta(x,y,ap,ap,bp,bHp,Adi,Adj,wlist)
#                D3=D_elem_delta(x,y,bp,bHp,ap,ap,Adi,Adj,wlist)
#                D4=D_elem_delta(x,y,bp,bHp,bp,bHp,Adi,Adj,wlist)
#                res+=c[i+1]*c[j+1]*(D1+D2+D3+D4)
#    return res

def DensityFuncD(x,c,alphaD,wlist):
    lengthD=len(alphaD)
    z1,z2,z2minus,eplus1,eplus2,eminus1,eminus2=svector.PionShift2()
    resD=0
    for i in range(lengthD):
        for j in range(lengthD):
            Adi=alphaD[i]; Adj=alphaD[j]
            D1=D_elem_delta(x,z1,z2,z2,z1,Adi,Adj,wlist)
            D2=D_elem_delta(x,z1,z2,eplus2,eminus1,Adi,Adj,wlist)
            D3=D_elem_delta(x,z1,eplus2,eminus2,z1,Adi,Adj,wlist)
            D4=D_elem_delta(x,z1,eplus2,z2minus,eminus1,Adi,Adj,wlist)
            D5=D_elem_delta(x,eplus1,eminus2,z2,z1,Adi,Adj,wlist)
            D6=D_elem_delta(x,eplus1,eminus2,eplus2,eminus1,Adi,Adj,wlist)
            D7=D_elem_delta(x,eplus1,z2minus,eminus2,z1,Adi,Adj,wlist)
            D8=D_elem_delta(x,eplus1,z2,z2,eminus1,Adi,Adj,wlist)
            resD+=c[i]*c[j]*(D1+D2+D3+D4+D5+D6+D7+D8)
    return 9*resD

def DensityFuncP(x,c,alphaP):
    lengthP=len(alphaP)
    alphaP=transform_list(alphaP)
    dimP=alphaP[0].shape[0]
    ap,bp,bHp=svector.create_pion_shift(dimP)
    resP=0
    for i in range(lengthP):
        for j in range(lengthP):
            Api=alphaP[i]; Apj=alphaP[j]
            P1=P_elem_delta(x,ap,ap,Api,Apj)
            P2=P_elem_delta(x,bp,bHp,Api,Apj)
            resP+=c[i+1]*c[j+1]*(P1+P2)
    return 3*resP

bmax=5
rmax=6*bmax
rmin=0.01*bmax
start=np.log(rmin)
stop=np.log(rmax)
ngausP=2
x=np.logspace(start,stop,num=3000,base=np.exp(1))


b=4
S=20 #Coupling strength parameters
mp=938.27 #Mev
mn=939.57
mpi0=134.98 
mpiC=139.57#Mev
mpi=(mpi0+mpiC)/2

params=np.zeros(2)
masses=np.zeros(2)
params[0]=b
params[1]=S
masses[0]=(mp+mn)/2
masses[1]=mpi

print('CALCULATING RESULTS FOR SINGLE PION DENSITY \n')

dictP=met.global_minP(ngausP,1,bmax,masses,params)
coords=dictP['coords'][-1]
cP=dictP['eigenvectors'][-1]

alphasP=[1/(b**2) for b in coords]

rhoP=DensityFuncP(x,cP,alphasP)

norm_test=quad(DensityFuncP,0,np.inf,args=(cP,alphasP))
NP,HP=mat.pion_test_1d(alphasP,masses,params)

Prob_comp=cP[1:].T@NP[1:,1:]@cP[1:]

print('Result of integral of P-wave Densityfunction:', norm_test[0], 'should be:', Prob_comp, '\n')

rhoP=np.reshape(rhoP,(3000,))

print('CALCULATING RESULTS FOR TWO PION DENSITY. \n')

ngausD=1
dictD=met.global_minD(ngausP,ngausD,1,3,bmax,masses,params,1)

coordsD=dictD['coords'][-1]
cD=dictD['eigenvectors'][-1]
massesD=dictD['masses']

wlist=tr.w_gen_2pion(massesD[0],massesD[1])
KD=met.K_gen(massesD[0],massesD[1],massesD[1])
alphasP2=[1/(b**2) for b in coordsD[:ngausP]]
alphasD2=[]
for i in range(0,len(coordsD[ngausP:]),3):
    A_add=tr.A_generate(coordsD[i+ngausP:i+ngausP+3],wlist)
    alphasD2.append(A_add)

c_int=cD[ngausP+1:]
cP2=cD[1:ngausP+1]
cP2_density=cD[:ngausP+1]

ND,HD=mat.PionTwo(alphasP2,alphasD2,params,massesD,wlist,KD)

Prob_comp2=c_int.T@ND[ngausP+1:,ngausP+1:]@c_int
print('Checking normalization of matrix elements:', cD.T@ND@cD, '\n')

norm_test2=quad(DensityFuncD,0,np.inf,args=(c_int,alphasD2,wlist))
print('Result of integral of D-wave Densityfunction:', norm_test2[0], 'should be:', Prob_comp2, '\n')

Prob_comp3=cP2.T@ND[1:ngausP+1,1:ngausP+1]@cP2
norm_test3=quad(DensityFuncP,0,np.inf,args=(cP2_density,alphasP2))

print('Result of integral of P-wave Densityfunction using D-wave function:', norm_test3[0], 'should be:', Prob_comp3, '\n')

rhoP2=DensityFuncP(x,cP2_density,alphasP2)
rhoP2=np.reshape(rhoP2,(3000,))

rhoD=DensityFuncD(x,c_int,alphasD2,wlist)
rhoD=np.reshape(rhoD,(3000,))

##Comparison of wavefunction squared in the one pion case.

def pWave(r,alphas,c):
    Pwavefunc=0
    for i in range(len(alphas)):
        for j in range(len(alphas)):
            Pwavefunc+=c[i]*c[j]*np.exp(-(alphas[i]+alphas[j])*r**2)
    return r**4*Pwavefunc

def pWave_test(r,alphas,c):
    Pwavefunc=0
    for i in range(len(alphas)):
        for j in range(len(alphas)):
            Pwavefunc+=c[i]*c[j]*np.exp(-(alphas[i]+alphas[j])*r**2)
    return r**4*Pwavefunc

intPwave=quad(pWave,0,np.inf, args=(alphasP2,cP2,))

print('Checking the norm of the P-wave wavefunction:', 3*4*np.pi*intPwave[0])

plt.figure(0)
plt.title('Density function compared to squared wavefunction')
plt.xlabel('x [fm]')
plt.ylabel(r'$\rho(x)$')
plt.plot(x,rhoP2, label='1 pion')
plt.plot(x,3*4*np.pi*np.reshape(pWave_test(x,alphasP2,cP2),x.shape), '--', label='P-wave squared')
#plt.plot(x,rhoD, label='2 pions')
#plt.plot(x,rhoD1, label='2 Pions - comparison')
plt.text(16,0.1,'Integral over norm squared=%s' %(np.around(3*4*np.pi*intPwave[0],3)))
plt.text(16,0.09, 'Integral of density function=%s' %(np.around(norm_test3[0],3)))
plt.legend()
plt.savefig('figures/Density_Comp_Wavefunc.pdf'.format(0),bbox_inches='tight')

plt.figure(1)
plt.title('Density function as a function of x')
plt.xlabel('x [fm]')
plt.ylabel(r'$\rho(x)$')
plt.plot(x,rhoP2, label='1 pion')
#plt.plot(x,3*4*np.pi*np.reshape(pWave(x,alphasP2,cP2),x.shape), '--', label='P-wave squared')
plt.plot(x,rhoD, label='2 pions')
plt.plot(x,rhoD+rhoP2, label='Sum of subsystems')
plt.text(15,0.030,'S=%s, b=%s' %(S,b))
plt.text(15,0.027,'One Pion contribution: %s' %(np.around(Prob_comp3[0][0],4)))
plt.text(15,0.024,'Two Pion contribution: %s' %(np.around(Prob_comp2[0][0],4)))
#plt.plot(x,rhoD1, label='2 Pions - comparison')
plt.legend()
plt.tight_layout()
plt.savefig('figures/Density_Comp_2pions.pdf'.format(0),bbox_inches='tight')

plt.show()
