import numpy as np
import matplotlib.pyplot as plt
import gauss_transformer as tr
from scipy.optimize import minimize
from scipy.linalg import eigh,det
import matrix_elements as mat

def K_gen(m1,m2,mP): ##TAKE THIS OUT AND MAKE IT A GENERAL METHOD. PionTwo should take K as input!
    M1=m1*m2/(m1+m2)
    M2=mP*(m1+m2)/(m1+m2+mP)
    K1=(197.3)**2/(2*M1)
    K2=(197.3)**2/(2*M2)
    A=np.array([[K1,0],[0,K2]])
    return A

def E_pionphoto(bs,masses,params): #Energy function
    A=[1/(b**2) for b in bs]
    N,H=mat.pion_test_1d(A,masses,params)
    E,c=eigh(H,N, subset_by_index=[0,0])
    print(E[0])
    c0=c
    return E,c0

def energy2Pion(bs,dimG,masses,params,w,w_COM,K):
    alphas=[] #Amount of parameters for P-wave Gaussians
    dim=len(w)
    bsP=bs[:dimG] #Parameters for P-wave Gaussians. 
    bsD=bs[dimG:] #Parameters for D-wave Gaussians. Remember that 1 Gaussian = 3 Parameters
    AP=[1/(b**2) for b in bsP]
    for i in range(0,len(bsD),dim):
        A=tr.A_generate(bsD[i:i+dim],w)
        alphas.append(A)
    N,H=mat.PionTwo(AP,alphas,params,masses,w_COM,K)
    E,c=eigh(H,N,subset_by_index=[0,0])
    return E,c

def energy2PionCoul(bs,dimG,masses,params,w,K):
    alphas=[] #Amount of parameters for P-wave Gaussians
    dim=len(w)
    bsP=bs[:dimG] #Parameters for P-wave Gaussians. 
    bsD=bs[dimG:] #Parameters for D-wave Gaussians. Remember that 1 Gaussian = 3 Parameters
    AP=[1/(b**2) for b in bsP]
    for i in range(0,len(bsD),dim):
        A=tr.A_generate(bsD[i:i+dim],w)
        alphas.append(A)
    N,H=mat.PionTwoCoulomb(AP,alphas,params,masses,w,K)
    E,c=eigh(H,N,subset_by_index=[0,0])
    return E,c

def minfuncD(bs,dimG,masses,params,w,w_COM,K):
    E,c=energy2Pion(bs,dimG,masses,params,w,w_COM,K)
    return E

def minfuncDCoul(bs,dimG,masses,params,w,K):
    E,c=energy2PionCoul(bs,dimG,masses,params,w,K)
    return E

def minfuncP(bs,masses,params):
    E,c=E_pionphoto(bs,masses,params)
    return E
    
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

def SVM_pseudo_test(ngaus,n_rand,dim,bmax,E_target,K,w):
    E_low=np.inf
    E_list2=[]
    bases2=np.array([])
    base_test2=np.array([])
    bij2=np.array([])
    gaussians=[]
    E_theo=[] #Initialization

    for i in range(ngaus):
        rand=np.random.rand(n_rand*dim)
        bij2=-np.log(rand)*bmax ##Generate random numbers and add them into a distribution to be used.
        for j in range(0,len(rand),dim):
            base_test2=np.append(base_test2,bij2[j:j+dim])
            E0=energyS(base_test2,K,w) #Calculate energy using test parameters
            if E0<=E_low:
                E_low=E0
                base_curr=np.copy(bij2[j:j+dim]) ##Add elements to current basis if energy is lower than E_low
            base_test2=base_test2[:-dim] #Takes out the tested element, leaving only the elements that are added to the base.
        bases2=np.append(bases2,base_curr) #Append current best basis to full basis.
        base_test2=np.append(base_test2,base_curr)
        E_list2.append(E_low) #Collect lowest energy at each iteration for plotting.
        gaussians.append(i)
        E_theo.append(E_target)
    return {'E_list': E_list2, 'gaussians': gaussians, 'E_theo': E_theo}

def SVM_pseudo(ngaus,n_rand,dim,bmax,E_target,params): #Test function to see if this method still works on the hydrogen anion.
    E_low=np.inf
    E_list2=[]
    bases2=np.array([])
    base_test2=np.array([])
    bij2=np.array([])
    gaussians=[]
    E_theo=[] #Initialization

    for i in range(ngaus):
        rand=np.random.rand(n_rand*dim)
        bij2=-np.log(rand)*bmax ##Generate random numbers and add them into a distribution to be used.
        for j in range(0,len(rand),dim):
            base_test2=np.append(base_test2,bij2[j:j+dim])
            E0=E_pionphoto(base_test2,params) #Calculate energy using test parameters
            if E0<=E_low:
                E_low=E0
                base_curr=np.copy(bij2[j:j+dim]) ##Add elements to current basis if energy is lower than E_low
            base_test2=base_test2[:-dim] #Takes out the tested element, leaving only the elements that are added to the base.
        bases2=np.append(bases2,base_curr) #Append current best basis to full basis.
        base_test2=np.append(base_test2,base_curr)
        E_list2.append(E_low) #Collect lowest energy at each iteration for plotting.
        gaussians.append(i)
        E_theo.append(E_target)
    return {'E_list': E_list2, 'gaussians': gaussians, 'E_theo': E_theo}

def SVM_pseudoD(ngaus,n_rand,dim,bmax,masses,params,w): #Test function to see if this method still works on the hydrogen anion.
    E_low=np.inf
    E_list2=[]
    c_list=[]
    bases2=np.array([])
    base_test2=np.array([])
    bij2=np.array([])
    gaussians=[]

    for i in range(ngaus):
        rand=np.random.rand(n_rand*dim)
        bij2=-np.log(rand)*bmax ##Generate random numbers and add them into a distribution to be used.
        for j in range(0,len(rand),dim):
            base_test2=np.append(base_test2,bij2[j:j+dim])
            E0,c0=energy2Pion(base_test2,dim,masses,params,w) #Calculate energy using test parameters
            if E0<=E_low:
                E_low=E0
                c_low=c0
                base_curr=np.copy(bij2[j:j+dim]) ##Add elements to current basis if energy is lower than E_low
            base_test2=base_test2[:-dim] #Takes out the tested element, leaving only the elements that are added to the base.
        bases2=np.append(bases2,base_curr) #Append current best basis to full basis.
        base_test2=np.append(base_test2,base_curr)
        E_list2.append(E_low) #Collect lowest energy at each iteration for plotting.
        c_list=np.append(c_low)
        gaussians.append(i)
    return E_list2, c_list, bases2

def global_minP(ngaus,dim,bmax,masses,params):
    E_list=[]
    gaussians=[]
    bs1=[] #Initialization
    coords=[]
    eigenvectors=[]
    E0S=0
    masses_min=np.copy(masses)
    n_calls=2

    for i in range(ngaus):
        hal=tr.halton(i+1,dim)
        bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.
        bs1=np.append(bs1,bs) #Add generated element to basis
        for j in range(n_calls):
            masses_min[0]=masses[0]-E0S
            print('Iteration of masses:', j)
            resS=minimize(minfuncP, bs1, args=(masses_min,params,), method="Nelder-Mead",options={'disp': True} ) #Optimize parameters
            E0S,C0S=E_pionphoto(resS.x,masses_min,params)
        E_list=np.append(E_list,E0S)
        coords.append(resS.x)
        eigenvectors.append(C0S)
        gaussians.append(i+1) #Energies are added to lists and collected in a dictionary
    return {'E_list': E_list, 'gaussians': gaussians, 'eigenvectors': eigenvectors, 'coords': coords, 'masses': masses_min}

#Jacobi is a trigger that alters between the two possible set of Jacobi coordinates we use.
def global_minD(ngausP,ngausD,dimP,dimG,bmax,masses,params,jacobi=None): #The index 0 mass of 'masses' must be nucleon mass.
    E_list=[]
    gaussians=[]
    bs1=[]
    coords=[]
    eigenvectors=[]
    E0=0
    masses_min=np.copy(masses)
    n_calls=2

    for i in range(ngausP+ngausD):
        if i<ngausP:
            hal=tr.halton(i+1,dimP)
            bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.
            bs1=np.append(bs1,bs)
            for j in range(n_calls):
                masses_min[0]=masses[0]-E0
                res=minimize(minfuncP, bs1, args=(masses_min,params,), method="Nelder-Mead")
                E0,c0=E_pionphoto(res.x,masses_min,params)
        else:
            hal=tr.halton(i+1,dimG)
            bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.
            bs1=np.append(bs1,bs)
            for j in range(n_calls):
                masses_min[0]=masses[0]-E0
                print(masses_min)
                w=tr.w_gen_2pion(masses_min[1],masses_min[1])
                w_COM=tr.w_gen_2pionCoM(masses_min[1],masses_min[1],masses_min[0])
                K=K_gen(masses_min[1],masses_min[1],masses_min[0])
                res=minimize(minfuncD,bs1,args=(ngausP,masses_min,params,w,w_COM,K,), method="Nelder-Mead")
                E0,c0=energy2Pion(res.x,ngausP,masses_min,params,w,w_COM,K)
            print('Energy:',E0)
        E_list=np.append(E_list,E0)
        coords.append(res.x)
        eigenvectors.append(c0)
        gaussians.append(i+1)
    return {'E_list': E_list, 'gaussians': gaussians, 'eigenvectors': eigenvectors, 'coords': coords, 'masses': masses_min}

def global_minDCoul(ngausP,ngausD,dimP,dimG,bmax,masses,params):
    E_list=[]
    gaussians=[]
    bs1=[]
    coords=[]
    eigenvectors=[]
    E0=0
    w=tr.w_gen_2pion(masses[1],masses[1])
    masses_min=np.copy(masses)

    for i in range(ngausP+ngausD):
        if i<ngausP:
            masses_min[0]=masses[0]-E0
            hal=tr.halton(i+1,dimP)
            bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.
            bs1=np.append(bs1,bs)
            res=minimize(minfuncP, bs1, args=(masses_min,params,), method="Nelder-Mead")
            E0,c0=E_pionphoto(res.x,masses_min,params)
        else:
            masses_min[0]=masses[0]-E0
            w=tr.w_gen_2pion(masses_min[1],masses_min[1])
            K=K_gen(masses_min[1],masses_min[1],masses_min[0])
            hal=tr.halton(i+1,dimG)
            bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.
            bs1=np.append(bs1,bs)
            res=minimize(minfuncDCoul,bs1,args=(ngausP,masses_min,params,w,K,), method="Nelder-Mead")
            E0,c0=energy2PionCoul(res.x,ngausP,masses_min,params,w,K)
        E_list=np.append(E_list,E0)
        coords.append(res.x)
        eigenvectors.append(c0)
        gaussians.append(i+1)
    return {'E_list': E_list, 'gaussians': gaussians, 'eigenvectors': eigenvectors, 'coords': coords, 'masses': masses_min}

def ParamMin(ParamGuess,bs,masses,w=None):
    resParams=minimize(lambda y: E_pionphoto(bs,masses,y), ParamGuess, method='BFGS')
    print('Minimizer: \n', resParams.message)
    return resParams.x

def plot_convergence(plot_params,idx,title,xlabel,ylabel,legendtxt,save_dest):
    plt.figure(idx)
    plt.plot(plot_params['gaussians'],plot_params['E_list'],marker='.')
    #plt.plot(plot_params['gaussians'],plot_params['E_theo'], '--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legendtxt)
    plt.savefig(save_dest.format(idx-1))
    plt.show()