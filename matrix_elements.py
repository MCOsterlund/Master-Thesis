import numpy as np
from scipy.linalg import eigh, det
from scipy.optimize import minimize
from svector_class import svector
import methods as met
import gauss_transformer as tr

def shift_dot(a,b,mat=None):
    n=a.shape[1]
    sum=0
    if not hasattr(mat, "__len__"):
        mat=np.identity(n)
    assert n==mat.shape[0], "ERROR! Matrix shape does not match number of shift vectors."
    for i in range(n):
        for j in range(n):
            dot=a[:,i]@b[:,j]
            sum +=mat[i,j]*dot
    return sum

def transform_list(alphas):
    g_new=[np.ones((1,1))*alphas[i] for i in range(len(alphas))]
    return g_new

def w_gen(dim, i,j):  
    if dim==1:
        w=np.ones((1,1))
        return w
    else:
        w=np.zeros((1,dim))
        w[i]=1
        w[j]=-1
        return w

def S_elem(A,B,K,w=None):
    dim=A.shape[0]
    coul=0
    D=A+B
    R=np.linalg.inv(D)
    M0=(np.pi**(dim)/np.linalg.det(D))**(3.0/2)
    trace=np.trace(B@K@A@R)
    if w!=None:
        for k in range(len(w)): ##Included to make the anion test run properly. Basically, script now handles taking a list of w's instead of just a single one. 
#The implementation is a bit hacky, since it only works if the list of w's has a length of two. 
            beta=1/(w[k].T@R@w[k])
            if k==2:
                coul+=2*np.sqrt(beta/np.pi)*M0 #Plus since we have two negative terms and one positive, meaning net one negative, that cancels minus.
            else:
                coul-=2*np.sqrt(beta/np.pi)*M0
        return M0,trace,coul
    else:
        return M0, trace

#Remember when calculating the S-wave wavefunction to multiply the trace with 6*M0 in order to get correct kinetic energy

def P_elem(a,b,A,B,K, w=None):
    D=A+B
    R=np.linalg.inv(D)
    M0,trace=S_elem(A,B,K)
    M1=1/2*((a*R*b))*M0 #Overlap
    kin=6*trace*M1 #Kinetic matrix elements
    kin+= (a*K*b)*M0
    kin+= -(a*(K@A@R)*b)*M0 
    kin+= -(a*(R@B@K)*b)*M0 
    kin+= (a*(R@B@K@A@R)*b)*M0  
    kin+= (a*(R@B@K@A@R)*b)*M0
    if w!=None:
        beta=1/(w.T@R@w) #Coulomb terms
        coul=2*np.sqrt(beta/np.pi)*M1
        coul+= -np.sqrt(beta/np.pi)*beta/3*(a*(R@w@w@R)*b)*M0 #Should only include terms for interacting particles.
        return M1, kin, coul
    else:
        return M1, kin

def D_elem(c,d,a,b,A,B,K,w=None): #Take the shift vectors as first input so i don't lose my mind.
    D=A+B
    R=np.linalg.inv(D)
    M0,trace=S_elem(A,B,K)
    kin=0; coul=0
    M2=1/4*((a*R*b)*(c*R*d) + (a*R*c)*(b*R*d) + (a*R*d)*(b*R*c))*M0 #Check
    kin+=6*trace*M2 #Check
    kin+=1/2*((a*K*c)*(b*R*d) + (a*K*d)*(b*R*c) + (b*K*c)*(a*R*d) + (b*K*d)*(a*R*c))*M0 #Check
    kin+=-1/2*((a*(R@B@K)*b)*(c*R*d)+(b*(R@B@K)*a)*(c*R*d))*M0 #Check
    kin+=-1/2*((c*(R@B@K)*a)*(b*R*d)+(c*(R@B@K)*b)*(a*R*d))*M0 #Check
    kin+=-1/2*((d*(R@B@K)*a)*(b*R*c)+(d*(R@B@K)*b)*(c*R*a))*M0 #Check
    kin+=-1/2*((c*(K@A@R)*a)*(b*R*d)+(c*(K@A@R)*b)*(a*R*d)+(c*(K@A@R)*d)*(a*R*b))*M0 #Check
    kin+=-1/2*((d*(K@A@R)*a)*(b*R*c)+(d*(K@A@R)*b)*(a*R*c)+(d*(K@A@R)*c)*(a*R*b))*M0 #Check
    kin+=1/2*((a*(R@B@K@A@R)*b)*(c*R*d)+(a*(R@B@K@A@R)*c)*(b*R*d)+(a*(R@B@K@A@R)*d)*(b*R*c))*M0 #Check
    kin+=1/2*((b*(R@B@K@A@R)*a)*(c*R*d)+(b*(R@B@K@A@R)*c)*(a*R*d)+(b*(R@B@K@A@R)*d)*(a*R*c))*M0 #Check
    kin+=1/2*((c*(R@B@K@A@R)*a)*(b*R*d)+(c*(R@B@K@A@R)*b)*(a*R*d)+(c*(R@B@K@A@R)*d)*(a*R*b))*M0 #Check
    kin+=1/2*((d*(R@B@K@A@R)*a)*(b*R*c)+(d*(R@B@K@A@R)*b)*(a*R*c)+(d*(R@B@K@A@R)*c)*(a*R*b))*M0 #Check
    if w!=None:
        beta=1/(w.T@R@w)
        coul+=2*np.sqrt(beta/np.pi)*M2 #Check
        coul+=-2*np.sqrt(beta/np.pi)*beta/3*1/4*((a*((R@w)@(w.T@R))*b)*(c*R*d)+(a*((R@w)@(w.T@R))*c)*(b*R*d)+(a*((R@w)@(w.T@R))*d)*(b*R*c)+(b*((R@w)@(w.T@R))*c)*(a*R*d)+(b*((R@w)@(w.T@R))*d)*(a*R*c)+(c*((R@w)@(w.T@R))*d)*(a*R*b))*M0 #Check
        coul+=2*np.sqrt(beta/np.pi)*(beta**2)*1/10*1/2*((a*((R@w)@(w.T@R))*b)*(c*((R@w)@(w.T@R))*d)+(a*((R@w)@(w.T@R))*c)*(b*((R@w)@(w.T@R))*d)+(a*((R@w)@(w.T@R))*d)*(b*((R@w)@(w.T@R))*c))*M0 #Check #Plus since we have two negative terms and one positive, meaning net one negative, that cancels minus.  
        return M2,kin,coul
    else:
        return M2, kin

def D_elem2Pion(a,b,c,d,A,B,K,w=None): #Take the shift vectors as first input so i don't lose my mind.
    D=A+B
    R=np.linalg.inv(D)
    M0,trace=S_elem(A,B,K)
    kin=0; coul=0
    M2=1/4*((a.T@R@b)*(c.T@R@d) + (a.T@R@c)*(b.T@R@d) + (a.T@R@d)*(b.T@R@c))*M0 #Check
    kin+=6*trace*M2 #Check
    kin+=1/2*((a.T@K@c)*(b.T@R@d) + (a.T@K@d)*(b.T@R@c) + (b.T@K@c)*(a.T@R@d) + (b.T@K@d)*(a.T@R@c))*M0 #Check
    kin+=-1/2*((a.T@(R@B@K)@b)*(c.T@R@d)+(b.T@(R@B@K)@a)*(c.T@R@d))*M0 #Check
    kin+=-1/2*((c.T@(R@B@K)@a)*(b.T@R@d)+(c.T@(R@B@K)@b)*(a.T@R@d))*M0 #Check
    kin+=-1/2*((d.T@(R@B@K)@a)*(b.T@R@c)+(d.T@(R@B@K)@b)*(c.T@R@a))*M0 #Check
    kin+=-1/2*((c.T@(K@A@R)@a)*(b.T@R@d)+(c.T@(K@A@R)@b)*(a.T@R@d)+(c.T@(K@A@R)@d)*(a.T@R@b))*M0 #Check
    kin+=-1/2*((d.T@(K@A@R)@a)*(b.T@R@c)+(d.T@(K@A@R)@b)*(a.T@R@c)+(d.T@(K@A@R)@c)*(a.T@R@b))*M0 #Check
    kin+=1/2*((a.T@(R@B@K@A@R)@b)*(c.T@R@d)+(a.T@(R@B@K@A@R)@c)*(b.T@R@d)+(a.T@(R@B@K@A@R)@d)*(b.T@R@c))*M0 #Check
    kin+=1/2*((b.T@(R@B@K@A@R)@a)*(c.T@R@d)+(b.T@(R@B@K@A@R)@c)*(a.T@R@d)+(b.T@(R@B@K@A@R)@d)*(a.T@R@c))*M0 #Check
    kin+=1/2*((c.T@(R@B@K@A@R)@a)*(b.T@R@d)+(c.T@(R@B@K@A@R)@b)*(a.T@R@d)+(c.T@(R@B@K@A@R)@d)*(a.T@R@b))*M0 #Check
    kin+=1/2*((d.T@(R@B@K@A@R)@a)*(b.T@R@c)+(d.T@(R@B@K@A@R)@b)*(a.T@R@c)+(d.T@(R@B@K@A@R)@c)*(a.T@R@b))*M0 #Check
    if w!=None:
        for i in range(len(w)):
            if i==0:
                beta=1/(w[i].T@R@w[i])
                coul+=2*np.sqrt(beta/np.pi)*M2 #Check
                coul+=-2*np.sqrt(beta/np.pi)*beta/3*1/4*((a.T@((R@w[i])@(w[i].T@R))@b)*(c.T@R@d)+(a.T@((R@w[i])@(w[i].T@R))@c)*(b.T@R@d)+(a.T@((R@w[i])@(w[i].T@R))@d)*(b.T@R@c)+(b.T@((R@w[i])@(w[i].T@R))@c)*(a.T@R@d)+(b.T@((R@w[i])@(w[i].T@R))@d)*(a.T@R@c)+(c.T@((R@w[i])@(w[i].T@R))@d)*(a.T@R@b))*M0 #Check
                coul+=2*np.sqrt(beta/np.pi)*(beta**2)*1/10*1/2*((a.T@((R@w[i])@(w[i].T@R))@b)*(c.T@((R@w[i])@(w[i].T@R))@d)+(a.T@((R@w[i])@(w[i].T@R))@c)*(b.T@((R@w[i])@(w[i].T@R))@d)+(a.T@((R@w[i])@(w[i].T@R))@d)*(b.T@((R@w[i])@(w[i].T@R))@c))*M0 #Check #Plus since we have two negative terms and one positive, meaning net one negative, that cancels minus.  
            else:
                beta=1/(w[i].T@R@w[i])
                coul-=2*np.sqrt(beta/np.pi)*M2 #Check
                coul-=-2*np.sqrt(beta/np.pi)*beta/3*1/4*((a.T@((R@w[i])@(w[i].T@R))@b)*(c.T@R@d)+(a.T@((R@w[i])@(w[i].T@R))@c)*(b.T@R@d)+(a.T@((R@w[i])@(w[i].T@R))@d)*(b.T@R@c)+(b.T@((R@w[i])@(w[i].T@R))@c)*(a.T@R@d)+(b.T@((R@w[i])@(w[i].T@R))@d)*(a.T@R@c)+(c.T@((R@w[i])@(w[i].T@R))@d)*(a.T@R@b))*M0 #Check
                coul-=2*np.sqrt(beta/np.pi)*(beta**2)*1/10*1/2*((a.T@((R@w[i])@(w[i].T@R))@b)*(c.T@((R@w[i])@(w[i].T@R))@d)+(a.T@((R@w[i])@(w[i].T@R))@c)*(b.T@((R@w[i])@(w[i].T@R))@d)+(a.T@((R@w[i])@(w[i].T@R))@d)*(b.T@((R@w[i])@(w[i].T@R))@c))*M0 #Check #Plus since we have two negative terms and one positive, meaning net one negative, that cancels minus. 
        return M2,kin,coul
    else:
        return M2, kin

#for k in range(len(w)): ##Included to make the anion test run properly. Basically, script now handles taking a list of w's instead of just a single one. 
#The implementation is a bit hacky, since it only works if the list of w's has a length of two. 
#            beta=1/(w[k].T@R@w[k])
#            if k==0:
#                coul+=2*np.sqrt(beta/np.pi)*M2 #Check
#                coul+=-2*np.sqrt(beta/np.pi)*beta/3*1/4*((a*((R@w[k])@(w[k].T@R))*b)*(c*R*d)+(a*((R@w[k])@(w[k].T@R))*c)*(b*R*d)+(a*((R@w[k])@(w[k].T@R))*d)*(b*R*c)+(b*((R@w[k])@(w[k].T@R))*c)*(a*R*d)+(b*((R@w[k])@(w[k].T@R))*d)*(a*R*c)+(c*((R@w[k])@(w[k].T@R))*d)*(a*R*b))*M0 #Check
#                coul+=2*np.sqrt(beta/np.pi)*(beta**2)*1/10*1/2*((a*((R@w[k])@(w[k].T@R))*b)*(c*((R@w[k])@(w[k].T@R))*d)+(a*((R@w[k])@(w[k].T@R))*c)*(b*((R@w[k])@(w[k].T@R))*d)+(a*((R@w[k])@(w[k].T@R))*d)*(b*((R@w[k])@(w[k].T@R))*c))*M0 #Check #Plus since we have two negative terms and one positive, meaning net one negative, that cancels minus.
#            else:
#                coul-=2*np.sqrt(beta/np.pi)*M2 #Check
#                coul-=-2*np.sqrt(beta/np.pi)*beta/3*1/4*((a*((R@w[k])@(w[k].T@R))*b)*(c*R*d)+(a*((R@w[k])@(w[k].T@R))*c)*(b*R*d)+(a*((R@w[k])@(w[k].T@R))*d)*(b*R*c)+(b*((R@w[k])@(w[k].T@R))*c)*(a*R*d)+(b*((R@w[k])@(w[k].T@R))*d)*(a*R*c)+(c*((R@w[k])@(w[k].T@R))*d)*(a*R*b))*M0 #Check
#                coul-=2*np.sqrt(beta/np.pi)*(beta**2)*1/10*1/2*((a*((R@w[k])@(w[k].T@R))*b)*(c*((R@w[k])@(w[k].T@R))*d)+(a*((R@w[k])@(w[k].T@R))*c)*(b*((R@w[k])@(w[k].T@R))*d)+(a*((R@w[k])@(w[k].T@R))*d)*(b*((R@w[k])@(w[k].T@R))*c))*M0 #Check #Plus since we have two negative terms and one positive, meaning net one negative, that cancels minus.

#I'm keeping Coulomb as a seperate method in the S_wave to D_wave functions, since these are kinda redundant now. 

def S_wave(alphas, K, w=None): 
    length=len(alphas)
    alphas=transform_list(alphas)
    kinetic=np.zeros((length,length))
    overlap=np.zeros((length,length))
    coulomb=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            if j<=i:
                A=alphas[i]; B=alphas[j]
                M0,trace,coul=S_elem(A,B,K,w)
                R=np.linalg.inv(A+B)
                overlap[i,j]=M0
                overlap[j,i]=overlap[i,j]
                kinetic[i,j]=6*trace*M0
                kinetic[j,i]=kinetic[i,j]
                coulomb[i,j]=coul
                coulomb[j,i]=coulomb[i,j]
    return overlap, kinetic, coulomb

def P_wave(alphas,K,w=None):
    length=len(alphas)
    alphas=transform_list(alphas)
    kinetic=np.zeros((length,length))
    overlap=np.zeros((length,length))
    coulomb=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            if j<=i:
                A=alphas[i]; B=alphas[j]
                dim=A.shape[0]
                a_vec=svector.create_shift(dim,p=1,d=None)
                a=a_vec*np.sqrt(1/A); b=a_vec*np.sqrt(1/B)
                D=A+B
                R=np.linalg.inv(D)
                M0,trace=S_elem(A,B,K)
                M1,kin,coul=P_elem(a,b,A,B,K,w)
                overlap[i,j]=M1
                overlap[j,i]=overlap[i,j]
                kinetic[i,j]=kin
                kinetic[j,i]=kinetic[i,j]
                coulomb[i,j]=coul
                coulomb[j,i]=coulomb[i,j]
    return overlap, kinetic, coulomb

def P_wave_comp(alphas,K,w=None):
    length=len(alphas)
    alphas=transform_list(alphas)
    dim=alphas[0].shape[0]
    kinetic=np.zeros((length,length))
    overlap=np.zeros((length,length))
    coulomb=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            if j<=i:
                a_vec=svector.create_shift(dim,p=1,d=None)
                a=a_vec*np.sqrt(1/alphas[i]); b=a_vec*np.sqrt(1/alphas[j])
                A=alphas[i]; B=alphas[j]
                D=A+B
                R=np.linalg.inv(D)
                M0=(np.pi**dim/np.linalg.det(D))**(3.0/2)
                trace=np.trace(B@K@A@R)
                M1=1/2*(a*R*b)*M0
                overlap[i,j]=M1
                overlap[j,i]=overlap[i,j]
                kinetic[i,j]=6*trace*M1
                kinetic[i,j]+= (a*K*b)*M0
                kinetic[i,j]+= -(a*(K@A@R)*b)*M0
                kinetic[i,j]+= -(b*(R@B@K)*a)*M0
                kinetic[i,j]+= (a*(R@B@K@A@R)*b)*M0
                kinetic[i,j]+= (b*(R@B@K@A@R)*a)*M0
                kinetic[j,i]=kinetic[i,j]
                if w!=None:
                    beta=1/(w.T@R@w)
                    coulomb[i,j]=2*np.sqrt(beta/np.pi)*M1
                    coulomb[i,j]+= -np.sqrt(beta/np.pi)*beta/3*(b*(R@w@w@R)*a)*M0
                    coulomb[j,i]=coulomb[i,j]
    return overlap, kinetic, coulomb

def D_wave(alphas,K,w=None):
    length=len(alphas)
    alphas=transform_list(alphas)
    dim=alphas[0].shape[0]
    kinetic=np.zeros((length,length))
    overlap=np.zeros((length,length))
    coulomb=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            if j<=i:
                a_vec,b_vec=svector.create_shift(dim,p=None,d=1)
                a=a_vec*np.sqrt(1/alphas[i]); b=b_vec*np.sqrt(1/alphas[j])
                c=a_vec*np.sqrt(1/alphas[i]); d=b_vec*np.sqrt(1/alphas[j])
                A=alphas[i]; B=alphas[j]
                D=A+B
                R=1/D
                M2,kin,coul=D_elem(c,d,a,b,A,B,K,w)
                overlap[i,j]=M2
                overlap[j,i]=overlap[i,j]
                kinetic[i,j]=kin
                kinetic[j,i]=kinetic[i,j]
                coulomb[i,j]=coul
                coulomb[j,i]=coulomb[i,j]
    return overlap, kinetic, coulomb

def pionOne(alphas, masses, params, w=None):
    b_w=params[0]
    S_w=params[1]
    mN=masses[0]
    mpi=masses[1] #Model parameters
    mNpi=mN*mpi/(mN+mpi) #Reduced mass of the Proton-Pion system
    K=np.identity(1)*((197.3)**2/(2*mNpi)) #Kinetic energy
    kap=1/b_w**2
    kap=np.identity(1)*kap #Kappa
    Amp=S_w/b_w #Initialization
    length=len(alphas)+1 #Make dimension one greater due to hardcoding parameter
    alphas=transform_list(alphas) #Transform my alphas into a list
    dim=alphas[0].shape[0] #Dimension for input into svector
    a,b,bH=svector.create_pion_shift(dim)
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
                    M1aW,kinaW=P_elem(a,a,kap,B,K)
                    M1bW,kinbW=P_elem(b,bH,kap,B,K)
                    overlap[i,j]=0
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=3*Amp*(M1aW+M1bW)
                    kinetic[j,i]=kinetic[i,j]
                else: ##Kinetic terms
                    A=alphas[i-1]; B=alphas[j-1]
                    if w==None: #If statement to include interactions without coulomb
                        M1aK, kinaK=P_elem(a,a,A,B,K,w)
                        M1bK, kinbK=P_elem(b,bH,A,B,K,w)
                        overlap[i,j]=3*(M1aK+M1bK)
                        overlap[j,i]=overlap[i,j]
                        kinetic[i,j]=(3*(kinaK+kinbK)+mpi*overlap[i,j])
                        kinetic[j,i]=kinetic[i,j]
                    else: #Includes Coulomb interaction
                        M1aK, kinaK, coula=P_elem(a,a,A,B,K,w)
                        M1bK, kinbK, coulb=P_elem(b,bH,A,B,K,w)
                        overlap[i,j]=3*(M1aK+M1bK)
                        overlap[j,i]=overlap[i,j]
                        kinetic[i,j]=(3*(kinaK+coula+ kinbK+coulb)+mpi*overlap[i,j])
                        kinetic[j,i]=kinetic[i,j]
    return overlap, kinetic

def pion_test_1d(alphas,masses,params):
    b_w=params[0]
    S_w=params[1]
    mN=masses[0]
    mpi=masses[1]
    mNpi=mN*mpi/(mN+mpi)
    K=(197.3)**2/(2*mNpi)
    kap=1/b_w**2
    kap=np.identity(1)*kap
    Amp=S_w/b_w #Initialization
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
                    kinetic[i,j]=3*Amp*3/2*1/(B+kap)*(np.pi/(B+kap))**(3/2)
                    kinetic[j,i]=kinetic[i,j]
                else: ##Kinetic terms
                    A=alphas[i-1]; B=alphas[j-1]
                    overlap[i,j]=3*3/2*1/(B+A)*(np.pi/(B+A))**(3/2)
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=3*K*15*A*B/((A+B)**2)*(np.pi/(A+B))**(3/2)+mpi*overlap[i,j]
                    kinetic[j,i]=kinetic[i,j]
    return overlap, kinetic

def photo_wavefunc(alphas,rs,cs):
    length=cs.shape[0]
    wavefunc=cs[0]**2
    for i in range(length):
        for j in range(length):
            wavefunc+=cs[i]*cs[j]*rs**4*np.exp(-(alphas[i]+alphas[j])*rs)
    return wavefunc

def PionTwo(alphasP, alphasD, params, masses, wlist,KD):
    mN=masses[0]
    mpi=masses[1] #Neutral pion mass
    b_w=params[0]
    S_w=params[1]
    mNpi=mN*mpi/(mN+mpi)
    KNpi=(197.3)**2/(2*mNpi)
    w_alpha=[wlist[0]]
    w_kap=[wlist[1]]
    kap=1/b_w**2
    kap=np.identity(1)*kap
    Amp=S_w/b_w #Initialization
    lengthP=len(alphasP)+1
    lengthD=len(alphasD)
    kinetic=np.zeros((lengthP+lengthD,lengthP+lengthD))
    overlap=np.zeros((lengthP+lengthD,lengthP+lengthD)) #Initialize matrices
    overlap[0,0]=1
    kinetic[0,0]=0 #Hardcoding parameters
    alphasP=transform_list(alphasP) #Transform my alphas into a list
    dimP=alphasP[0].shape[0] #Dimension for input into svector
    aP,bP,bHP=svector.create_pion_shift(dimP) #P-wave shifts
    z1,z2,z2minus,eplus1,eplus2,eminus1,eminus2=svector.PionShift2()
    for i in range(lengthP+lengthD):
        for j in range(lengthP+lengthD):
            if j<=i:
                if i==0 and j==0:
                    continue
                elif j==0 and i<lengthP: #First creation terms (P-wave)
                    B=alphasP[i-1]
                    M1aW,kinaW=P_elem(aP,aP,B,kap,KNpi*np.identity(1))
                    M1bW,kinbW=P_elem(bP,bHP,B,kap,KNpi*np.identity(1))
                    overlap[i,j]=0
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=3*Amp*(M1aW+M1bW)
                    kinetic[j,i]=kinetic[i,j] 
                elif j>0 and j<lengthP and i>=lengthP: #Second creation term
                    AP=tr.A_generate(1/np.sqrt(alphasP[j-1]),w_alpha); AD=alphasD[i-(lengthP)]
                    kap2=tr.A_generate(b_w,w_kap)
                    M21W,kinW1=D_elem2Pion(z1,z2,z2,z1,AD,AP+kap2, KD)
                    M22W,kinW2=D_elem2Pion(z1,z2,eplus2,eminus1,AD,AP+kap2, KD)
                    M23W,kinW3=D_elem2Pion(z1,eplus2,eminus2,z1,AD,AP+kap2, KD)
                    M24W,kinW4=D_elem2Pion(z1,eplus2,z2minus,eminus1,AD,AP+kap2,KD)
                    M25W,kinW5=D_elem2Pion(eplus1,eminus2,z2,z1,AD,AP+kap2,KD)
                    M26W,kinW6=D_elem2Pion(eplus1,eminus2,eplus2,eminus1,AD,AP+kap2,KD)
                    M27W,kinW7=D_elem2Pion(eplus1,z2minus,eminus2,z1,AD,AP+kap2,KD)
                    M28W,kinW9=D_elem2Pion(eplus1,z2,z2,eminus1,AD,AP+kap2,KD)
                    overlap[i,j]=0
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=9*(M21W+M22W+M23W+M24W+M25W+M26W+M27W+M28W) #Coulomb term omitted for now. Should be included later on.
                    kinetic[j,i]=kinetic[i,j]
                elif j!=0 and j<lengthP and i<lengthP: #KPpi
                    A=alphasP[i-1]; B=alphasP[j-1]
                    M1aK, kinaK=P_elem(aP,aP,A,B,KNpi*np.identity(1))
                    M1bK, kinbK=P_elem(bHP,bP,A,B,KNpi*np.identity(1))
                    overlap[i,j]=3*(M1aK+M1bK)
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=(3*(kinaK+kinbK)+mpi*overlap[i,j])
                    kinetic[j,i]=kinetic[i,j]
                elif i>=lengthP and j>=lengthP: #KPpipi
                    A=alphasD[i-lengthP]; B=alphasD[j-lengthP]
                    M1,kin1=D_elem2Pion(z1,z2,z2,z1,A,B, KD)
                    M2,kin2=D_elem2Pion(z1,z2,eplus2,eminus1,A,B, KD)
                    M3,kin3=D_elem2Pion(z1,eplus2,eminus2,z1,A,B, KD)
                    M4,kin4=D_elem2Pion(z1,eplus2,z2minus,eminus1,A,B,KD)
                    M5,kin5=D_elem2Pion(eplus1,eminus2,z2,z1,A,B,KD)
                    M6,kin6=D_elem2Pion(eplus1,eminus2,eplus2,eminus1,A,B,KD)
                    M7,kin7=D_elem2Pion(eplus1,z2minus,eminus2,z1,A,B,KD)
                    M8,kin8=D_elem2Pion(eplus1,z2,z2,eminus1,A,B,KD)
                    overlap[i,j]=9*(M1+M2+M3+M4+M5+M6+M7+M8)
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=9*(kin1+kin2+kin3+kin4+kin5+kin6+kin7+kin8) + 2*mpi*overlap[i,j] 
                    kinetic[j,i]=kinetic[i,j]
    return overlap, kinetic
                
def PionTwo_Test(alphasP, alphasD, params, masses,wlist,KD):
    m_P=masses[0]
    mpi0=masses[1] #Neutral pion mass
    b_w=params[0]
    S_w=params[1]
    mpi0P=m_P*mpi0/(m_P+mpi0)
    KNpi=(197.3)**2/(2*mpi0P)
    w_alpha=[wlist[1]]
    w_kap=[wlist[2]]
    kap=1/b_w**2
    kap=np.identity(1)*kap
    Amp=S_w/b_w #Initialization
    lengthP=len(alphasP)+1
    lengthD=len(alphasD)
    kinetic=np.zeros((lengthP+lengthD,lengthP+lengthD))
    overlap=np.zeros((lengthP+lengthD,lengthP+lengthD)) #Initialize matrices
    overlap[0,0]=1
    kinetic[0,0]=0 #Hardcoding parameters
    alphasP=transform_list(alphasP) #Transform my alphas into a list
    dimP=alphasP[0].shape[0] #Dimension for input into svector
    aP,bP,bHP=svector.create_pion_shift(dimP)
    for i in range(lengthP+lengthD):
        for j in range(lengthP+lengthD):
            if j<=i:
                if i==0 and j==0:
                    continue
                elif j==0 and i<lengthP: #First creation terms (P-wave)
                    B=alphasP[i-1]
                    M1aW,kinaW=P_elem(aP,aP,B,kap,KNpi*np.identity(1))
                    M1bW,kinbW=P_elem(bP,bHP,B,kap,KNpi*np.identity(1))
                    overlap[i,j]=0
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=3*Amp*(M1aW+M1bW)
                    kinetic[j,i]=kinetic[i,j] 
                elif j>0 and j<lengthP and i>=lengthP: #Second creation term
                    AP=tr.A_generate(1/np.sqrt(alphasP[j-1]),w_alpha); AD=alphasD[i-(lengthP)]
                    kap2=tr.A_generate(b_w,w_kap)
                    D2W=AP+AD+kap2
                    D2W11=D2W[0,0]
                    D2W22=D2W[1,1]
                    D2W12=D2W[0,1]
                    overlap[i,j]=0
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=9*3/4*np.pi**3*(3*D2W11*D2W22+2*D2W12**2)/(det(D2W)**(7/2)) #Coulomb term omitted for now. Should be included later on.
                    kinetic[j,i]=kinetic[i,j]
                elif j!=0 and j<lengthP and i<lengthP: #KPpi
                    A=alphasP[i-1]; B=alphasP[j-1]
                    M1aK, kinaK=P_elem(aP,aP,A,B,KNpi*np.identity(1))
                    M1bK, kinbK=P_elem(bHP,bP,A,B,KNpi*np.identity(1))
                    overlap[i,j]=3*(M1aK+M1bK)
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=(3*(kinaK+kinbK)+mpi0*overlap[i,j])
                    kinetic[j,i]=kinetic[i,j]
                elif i>=lengthP and j>=lengthP: #KPpipi
                    A=alphasD[i-lengthP]; B=alphasD[j-lengthP]
                    D=A+B
                    Ktot=0
                    A11=A[0,0]
                    A22=A[1,1]
                    A12=A[0,1]
                    B11=B[0,0]
                    B22=B[1,1]
                    B12=B[0,1]
                    D11=D[0,0]
                    D22=D[1,1]
                    D12=D[0,1]
                    K11=KD[0,0]
                    K22=KD[1,1]
                    Ktot+=6*np.pi**2/4*(D11)*K11/(det(D)**(5/2))
                    Ktot+=6*np.pi**2/4*(5*D11*(B12*K11*A12+A22*B22*K22))/(det(D)**(7/2))
                    Ktot+=6*np.pi**2/4*(5*D11*K11*A12*D12)/(det(D)**(7/2))
                    Ktot+=6*np.pi**2/4*(5*D11*K11*B12*D12)/(det(D)**(7/2))
                    Ktot+=6*np.pi**2/4*(35*D11*D12**2*(B12*K11*A12+A22*B22*K22))/(det(D)**(9/2))
                    overlap[i,j]=9*3/4*(np.pi**3)*(3*D11*D22+2*D12**2)/(det(D)**(7/2))
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]= 2*mpi0*overlap[i,j] 
                    kinetic[j,i]=kinetic[i,j]
    return overlap, kinetic

def PionTwoCoulomb(alphasP, alphasD, params, masses,wlist,KD):
    mN=masses[0]
    mpi=masses[1] #Neutral pion mass
    b_w=params[0]
    S_w=params[1]
    mNpi=mN*mpi/(mN+mpi)
    KNpi=(197.3)**2/(2*mNpi)
    w_alpha=[wlist[1]]
    w_kap=[wlist[2]]
    kap=1/b_w**2
    kap=np.identity(1)*kap
    Amp=S_w/b_w #Initialization
    lengthP=len(alphasP)+1
    lengthD=len(alphasD)
    kinetic=np.zeros((lengthP+lengthD,lengthP+lengthD))
    overlap=np.zeros((lengthP+lengthD,lengthP+lengthD)) #Initialize matrices
    overlap[0,0]=1
    kinetic[0,0]=0 #Hardcoding parameters
    alphasP=transform_list(alphasP) #Transform my alphas into a list
    dimP=alphasP[0].shape[0] #Dimension for input into svector
    aP,bP,bHP=svector.create_pion_shift(dimP)
    z1,z2,z2minus,eplus1,eplus2,eminus1,eminus2=svector.PionShift2()
    for i in range(lengthP+lengthD):
        for j in range(lengthP+lengthD):
            if j<=i:
                if i==0 and j==0:
                    continue
                elif j==0 and i<lengthP: #First creation terms (P-wave)
                    B=alphasP[i-1]
                    M1aW,kinaW=P_elem(aP,aP,B,kap,KNpi*np.identity(1))
                    M1bW,kinbW=P_elem(bP,bHP,B,kap,KNpi*np.identity(1))
                    overlap[i,j]=0
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=3*Amp*(M1aW+M1bW)
                    kinetic[j,i]=kinetic[i,j] 
                elif j>0 and j<lengthP and i>=lengthP: #Second creation term
                    AP=tr.A_generate(1/np.sqrt(alphasP[j-1]),w_alpha); AD=alphasD[i-(lengthP)]
                    kap2=tr.A_generate(b_w,w_kap)
                    M21W,kinW1=D_elem2Pion(z1,z2,z2,z1,AD,AP+kap2, KD)
                    M22W,kinW2=D_elem2Pion(z1,z2,eplus2,eminus1,AD,AP+kap2, KD)
                    M23W,kinW3=D_elem2Pion(z1,eplus2,eminus2,z1,AD,AP+kap2, KD)
                    M24W,kinW4=D_elem2Pion(z1,eplus2,z2minus,eminus1,AD,AP+kap2,KD)
                    M25W,kinW5=D_elem2Pion(eplus1,eminus2,z2,z1,AD,AP+kap2,KD)
                    M26W,kinW6=D_elem2Pion(eplus1,eminus2,eplus2,eminus1,AD,AP+kap2,KD)
                    M27W,kinW7=D_elem2Pion(eplus1,z2minus,eminus2,z1,AD,AP+kap2,KD)
                    M28W,kinW9=D_elem2Pion(eplus1,z2,z2,eminus1,AD,AP+kap2,KD)
                    overlap[i,j]=0
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=9*(M21W+M22W+M23W+M24W+M25W+M26W+M27W+M28W) #Coulomb term omitted for now. Should be included later on.
                    kinetic[j,i]=kinetic[i,j]
                elif j!=0 and j<lengthP and i<lengthP: #KPpi
                    A=alphasP[i-1]; B=alphasP[j-1]
                    M1aK, kinaK=P_elem(aP,aP,A,B,KNpi*np.identity(1))
                    M1bK, kinbK=P_elem(bHP,bP,A,B,KNpi*np.identity(1))
                    overlap[i,j]=3*(M1aK+M1bK)
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=(3*(kinaK+kinbK)+mpi*overlap[i,j])
                    kinetic[j,i]=kinetic[i,j]
                elif i>=lengthP and j>=lengthP: #KPpipi
                    A=alphasD[i-lengthP]; B=alphasD[j-lengthP]
                    M1,kin1,coul1=D_elem2Pion(z1,z2,z2,z1,A,B, KD,wlist)
                    M2,kin2,coul2=D_elem2Pion(z1,z2,eplus2,eminus1,A,B, KD,wlist)
                    M3,kin3,coul3=D_elem2Pion(z1,eplus2,eminus2,z1,A,B, KD,wlist)
                    M4,kin4,coul4=D_elem2Pion(z1,eplus2,z2minus,eminus1,A,B,KD,wlist)
                    M5,kin5,coul5=D_elem2Pion(eplus1,eminus2,z2,z1,A,B,KD,wlist)
                    M6,kin6,coul6=D_elem2Pion(eplus1,eminus2,eplus2,eminus1,A,B,KD,wlist)
                    M7,kin7,coul7=D_elem2Pion(eplus1,z2minus,eminus2,z1,A,B,KD,wlist)
                    M8,kin8,coul8=D_elem2Pion(eplus1,z2,z2,eminus1,A,B,KD,wlist)
                    overlap[i,j]=9*(M1+M2+M3+M4+M5+M6+M7+M8)
                    overlap[j,i]=overlap[i,j]
                    kinetic[i,j]=9*(kin1+kin2+kin3+kin4+kin5+kin6+kin7+kin8) + 2*mpi*overlap[i,j] + 4*(coul1+coul2+coul3+coul4+coul5+coul6+coul7+coul8)
                    kinetic[j,i]=kinetic[i,j]
    return overlap, kinetic