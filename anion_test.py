import numpy as np
import matplotlib.pyplot as plt
import matrix_elements as mat
import gauss_transformer as tr
from scipy.optimize import minimize
from scipy.linalg import eigh, det
import methods as met

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
    E0=np.amin(E)
    return E0

b1=7
E_list=[]
gaussians=[]
E_theoS=[]
bij=np.array([])
E_S=-0.527

#print("---------QUASI-RANDOM METHOD---------")
#
#E_low=np.inf
#bases=np.array([])
#base_test=np.array([])
#for i in range(25):
#    hal=tr.halton(i+1,15*len(w_trans))
#    bij=-np.log(hal)*b1
#    for j in range(0,len(hal),len(w_trans)):
#        base_test=np.append(base_test,bij[j:j+len(w_trans)])
#        E0=energyS(base_test,K_trans,w_trans)
#        if E0<=E_low:
#            E_low=E0
#            base_curr=np.copy(bij[j:j+len(w_trans)])
#        base_test=base_test[:-len(w_trans)]
#    bases=np.append(bases,base_curr)
#    base_test=np.append(base_test,base_curr)
#    E_list.append(E_low)
#    print(E_low)
#    gaussians.append(i)
#    E_theoS.append(E_S)
#
#print("Best convergent numerical value:", E_list[-1])
#print("Theoretical value:", E_S)
#print("Difference:", np.abs(E_list[-1]-E_S))
#
#print("---------QUASI-RANDOM METHOD W. REFINEMENT---------")
#
#bases_ref=np.copy(bases)
#E_ref=E_list[-1]
#E_list_ref=[]
#for i in range(len(bases_ref)-len(w_trans)):
#    rand_ref=np.random.rand(200*len(w_trans))
#    bij_ref=-np.log(rand_ref)*b1
#    for j in range(0, len(rand_ref),len(w_trans)):
#        bases_ref[i:i+len(w_trans)]=bij_ref[j:j+len(w_trans)]
#        E_test=energyS(bases_ref,K_trans,w_trans)
#        if E_test<E_ref:
#            E_ref=E_test
#            bases[i:i+len(w_trans)]=bij_ref[j:j+len(w_trans)]
#    bases_ref=np.copy(bases)
#    E_list_ref.append(E_ref)
#    print('E_ref:', E_ref)
#
#print('Energy after refinement:', E_ref)
#print('Difference in energy from before refinement:', np.abs(E_ref-E_list[-1]))
#print('Difference from target value:', np.abs(E_ref-E_S))

#print("---------PSEUDO-RANDOM METHOD (RANDOM GUESSING)---------")
#
#E_low=np.inf
#E_list2=[]
#bases2=np.array([])
#base_test2=np.array([])
#bij2=np.array([])
#
#for i in range(20):
#    rand=np.random.rand(400*len(w_trans))
#    bij2=-np.log(rand)*b1
#    for j in range(0,len(rand),len(w_trans)):
#        base_test2=np.append(base_test2,bij2[j:j+len(w_trans)])
#        E0=energyS(base_test2,K_trans,w_trans)
#        if E0<=E_low:
#            E_low=E0
#            base_curr=np.copy(bij2[j:j+len(w_trans)])
#        base_test2=base_test2[:-len(w_trans)]
#    bases2=np.append(bases2,base_curr)
#    base_test2=np.append(base_test2,base_curr)
#    E_list2.append(E_low)
#    gaussians.append(i)
#    E_theoS.append(E_S)
#
#
#print("Best convergent numerical value:", E_list2[-1])
#print("Theoretical value:", E_S)
#print("Difference:", np.abs(E_list2[-1]-E_S))

print("----------TEST METHOD SCRIPT--------------")

E_list_test=met.SVM_pseudo_test(20,400, len(w_trans), b1, E_S, K_trans, w_trans)

print("Best convergent numerical value:", E_list_test['E_list'][-1])
print("Theoretical value:", E_S)
print("Difference:", np.abs(E_S-E_list_test['E_list'][-1]))

print("Full list:", E_list_test['E_list'])

#print("---------PSEUDO-RANDOM METHOD W. REFINEMENT---------")
#print("Compare value before and after refining.")
#print("Energy before refining", E_list2[-1])
#bases_ref=np.copy(bases2)
#E_ref=E_list2[-1]
#E_list_ref=[]
#for i in range(len(bases_ref)-len(w_trans)):
#    rand_ref=np.random.rand(200*len(w_trans))
#    bij_ref=-np.log(rand_ref)*b1
#    for j in range(0,len(rand_ref),len(w_trans)):
#        bases_ref[i:i+len(w_trans)]=bij_ref[j:j+len(w_trans)]
#        E_test=energyS(bases_ref,K_trans,w_trans)
#        if E_test<E_ref:
#            E_ref=E_test
#            bases2[i:i+len(w_trans)]=bij_ref[j:j+len(w_trans)]
#    bases_ref=np.copy(bases2)
#    E_list_ref.append(energyS(bases2,K_trans,w_trans))
#    print('E_ref', E_ref)
#print('Energy after refinement:', E_ref)
#print('Difference in energy from before refinement:', np.abs(E_ref-E_list2[-1]))
#print('Difference from target value:', np.abs(E_ref-E_S))

#print("---------PSEUDO-RANDOM METHOD W. FINAL MINIMIZATION---------")
#print("Compare value before and after minimization.")
#print("Energy before minimization", E_list2[-1])
#
#resS=minimize(energyS, bases2, args=(K_trans,w_trans,), method="BFGS")
#E_min=energyS(resS.x,K_trans,w_trans)
#print("Energy after minimization:", E_min)
#print("Difference from before minimization:", np.abs(E_min-E_list2[-1]))
#print("Difference from wanted value:", np.abs(E_min-E_S))

plt.figure(1)
plt.plot(gaussians,E_list2,marker='.')
plt.plot(gaussians,E_list_ref, marker='.')
plt.plot(gaussians,E_theoS, '--')
plt.title('S-wave convergence of Hydrogen anion')
plt.xlabel('Number of Gaussians')
plt.ylabel('Energy [Hartree]')
plt.legend(['Pseudo-random','Refinement process','Theoretical value'])
plt.savefig('figures/anion_groundstate_SVM.pdf'.format(0))
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