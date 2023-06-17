import numpy as np
import matplotlib.pyplot as plt
import matrix_elements as mat
import gauss_transformer as tr
from scipy.optimize import minimize
from scipy.linalg import eigh, det

w_list=tr.w_gen_3()
m_list=np.array([np.inf, 1, 1])
K=np.array([[0,0,0],[0,1/2,0],[0,0,1/2]])
J,U=tr.jacobi_transform(m_list)
K_trans=J@K@J.T
w_trans=[U.T @ w_list[i] for i in range(len(w_list))]

def energyS(bij,K,w):
    alphas=[]
    dim=len(w)
    for i in range(0,len(bij),dim):
        A=tr.A_generate(bij[i:i+dim],w)
        alphas.append(A)
    N, kinetic, coulomb=mat.S_wave(alphas,K,w)
    H=kinetic+coulomb
    E=eigh(H,N, eigvals_only='true')
    E0=jnp.amin(E)
    return E0

bmax=7
ngaus=15
E_list=[]
gaussians=[]
E_theoS=[]
bij=np.array([])
bs1=[]
E_S=-0.527

for i in range(ngaus):
    hal=tr.halton(i+1,len(w_trans))
    bs=-np.log(hal)*bmax
    bs1=np.append(bs1,bs)
    resS=minimize(energyS, bs1, args=(K_trans,w_trans,), method="Nelder-Mead")
    E0S=energyS(resS.x,K_trans,w_trans)
    E_list=np.append(E_list,E0S)
    E_theoS.append(E_S)
    gaussians.append(i+1)
    print(E0S)

plt.figure(1)
plt.plot(gaussians,E_list, marker='.')
plt.plot(gaussians,E_theoS, '--')
plt.title('S-wave convergence of Hydrogen anion')
plt.xlabel('Number of Gaussians')
plt.ylabel('Energy [Hartree]')
plt.legend(['Numerical estimate', 'Theoretical value'])
plt.savefig('figures/anion_groundstate_Optimizer.pdf'.format(0))
plt.show()

##Trialing with a system of a positron and two electrons

#w_list2=tr.w_gen_3()
#m_list2=np.array([1,1,1])
#K2=np.array([[1/2,0,0],[0,1/2,0],[0,0,1/2]])
#J2,U2=tr.jacobi_transform(m_list2)
#K_trans2=J2@K2@J2.T
#w_trans2=[U2.T @ w_list2[i] for i in range(len(w_list2))]
#
#b2=7
#E_list2=[]
#gaussians2=[]
#E_theoS2=[]
#bij2=np.array([])
#E_S2=-0.262005
#
#E_low2=np.inf
#bases2=np.array([])
#base_test2=np.array([])
#for i in range(100):
#    hal2=tr.halton(i+1,7*len(w_trans2))
#    bij2=-np.log(hal2)*b2
#    for j in range(0,len(hal2),len(w_trans2)):
#        base_test2=np.append(base_test2,bij2[j:j+len(w_trans2)])
#        E02=energyS(base_test2,K_trans2,w_trans2)
#        if E02<=E_low2:
#            E_low2=E02
#            base_curr2=np.copy(bij2[j:j+len(w_trans2)])
#        base_test2=base_test2[:-len(w_trans2)]
#    bases2=np.append(bases2,base_curr2)
#    base_test2=np.append(base_test2,base_curr2)
#    E_list2.append(E_low2)
#    gaussians2.append(i+1)
#    E_theoS2.append(E_S2)
#
#print("Best convergent numerical value:", E_list2[-1])
#print("Theoretical value:", E_S2)
#print("Difference:", np.abs(E_list2[-1]-E_S2))
#
#plt.figure(2)
#plt.plot(gaussians2,E_list2, marker='.')
#plt.plot(gaussians2,E_theoS2, '--')
#plt.title('S-wave convergence of Positron and two Electron System')
#plt.xlabel('Number of Gaussians')
#plt.ylabel('Energy [Hartree]')
#plt.legend(['Numerical result', 'Theoretical value'])
#plt.savefig('electron_positron_SVM.pdf'.format(0))
#plt.show()