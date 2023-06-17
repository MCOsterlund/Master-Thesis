import numpy as np
import matplotlib.pyplot as plt
import gauss_transformer as tr
from scipy.optimize import minimize
from scipy.linalg import eigh,det
import matrix_elements as mat

def E_pionphoto(bs,masses,params): #Energy function
    A=[1/(b**2) for b in bs]
    N,H=mat.pion_test_1d(A,masses,params)
    E,c=eigh(H,N, subset_by_index=[0,0])
    c0=c
    return E,c0

def energy2Pion(bs,ngausP,masses,params,w):
    alphas=[] #Amount of parameters for P-wave Gaussians
    dim=len(w)
    bsP=bs[:ngausP] #Parameters for P-wave Gaussians. 
    bsD=bs[ngausP:] #Parameters for D-wave Gaussians. Remember that 1 Gaussian = 3 Parameters
    AP=[1/(b**2) for b in bsP]
    for i in range(0,len(bsD),dim):
        A=tr.A_generate(bsD[i:i+dim],w)
        alphas.append(A)
    N,H=mat.PionTwo(AP,alphas,params,masses)
    E,c=eigh(H,N,subset_by_index=[0,0])
    return E,c

def minfuncD(bs,dimG,masses,params,w):
    E,c=energy2Pion(bs,dimG,masses,params,w)
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

def SVM_pseudoD(ngaus,n_rand,dim,dimG,bmax,masses,params,w): #Test function to see if this method still works on the hydrogen anion.
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
            E0,c0=energy2Pion(base_test2,dimG,masses,params,w) #Calculate energy using test parameters
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

    for i in range(ngaus):
        masses_min[0]=masses[0]-E0S
        hal=tr.halton(i+1,dim)
        bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.
        bs1=np.append(bs1,bs) #Add generated element to basis
        resS=minimize(minfuncP, bs1, args=(masses_min,params,), method="Nelder-Mead") #Optimize parameters
        E0S,C0S=E_pionphoto(resS.x,masses_min,params)
        E_list=np.append(E_list,E0S)
        coords.append(resS.x)
        eigenvectors.append(C0S)
        gaussians.append(i+1) #Energies are added to lists and collected in a dictionary
    return {'E_list': E_list, 'gaussians': gaussians, 'eigenvectors': eigenvectors, 'coords': coords, 'masses': masses_min}

 
def global_minD(ngausP,ngausD,dimP,dimG,bmax,masses,params):
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
            hal=tr.halton(i+1,dimG)
            bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.
            bs1=np.append(bs1,bs)
            res=minimize(minfuncD, bs1, args=(ngausP,masses_min,params,w,), method="Nelder-Mead")
            E0,c0=energy2Pion(bs1,ngausP,masses,params,w)
        coords.append(res.x)
        E_list=np.append(E_list,E0)
        eigenvectors.append(c0)
        gaussians.append(i+1)
    return {'E_list': E_list, 'gaussians': gaussians, 'eigenvectors': eigenvectors, 'coords': coords, 'masses': masses_min}

def pseudo_minD(ngausP,ngausD,dimP,dimG,bmax,masses,params):
    E_list=[]
    gaussians=[]
    bs1=[]
    coords=[]
    eigenvectors=[]
    E0=0
    w=tr.w_gen_2pion(masses[1],masses[1])
    masses_min=np.copy(masses)
    dim=3
    E_low=np.inf

    for i in range(ngausP+ngausD):
        if i<ngausP:
            masses_min[0]=masses[0]-E0
            hal=tr.halton(i+1,dimP)
            bs=-np.log(hal)*bmax #Generate halton sequence of numbers and add to exponential distribution.
            bs1=np.append(bs1,bs)
            res=minimize(minfuncP, bs1, args=(masses_min,params,), method="Nelder-Mead")
            E0,c0=E_pionphoto(res.x,masses_min,params)
            E_list=np.append(E_list,E0)
            coords.append(res.x)
            eigenvectors.append(c0)
        else:
            masses_min[0]=masses[0]-E0
            rand=np.random.rand(400*dim)
            bij2=-np.log(rand)*bmax ##Generate random numbers and add them into a distribution to be used.
            for j in range(0,len(rand),dim):
                base_test2=np.append(bs1,bij2[j:j+dim])
                E0,c0=energy2Pion(base_test2,dimG,masses,params,w) #Calculate energy using test parameters
                if E0<=E_low:
                    E_low=E0
                    c_low=c0
                    base_curr=np.copy(bij2[j:j+dim]) ##Add elements to current basis if energy is lower than E_low
                base_test2=bs1[:-dim]
            E_list=np.append(E_list,E0)
            coords.append(base_curr)
            bs1=np.append(bs1,base_curr)
            eigenvectors.append(c_low)
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