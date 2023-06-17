import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import eigh
import matrix_elements as mat
from svector_class import svector

def jacobi_transform(m_list):
    dim=m_list.shape[0]
    J=np.zeros((dim,dim))
    for i in range(dim):
        sum_m=np.sum(m_list[:i+1])
        for j in range(dim):
            if j==i+1:
                J[i,j]=-1
            elif i+1<j:
                J[i,j]=0
            else:
                J[i,j]=m_list[j]/sum_m
            if np.isnan(J[i,j]):
                J[i,j]=1
    U=np.linalg.inv(J)
    if 1<dim:
        U=np.delete(U,dim-1,1)
        J=np.delete(J,dim-1,0)
    return J,U

def w_gen_3():
    w1=np.zeros((3))
    w2=np.zeros((3))
    w3=np.zeros((3))
    w1[0]=1
    w1[1]=-1
    w2[0]=1
    w2[2]=-1
    w3[1]=1
    w3[2]=-1
    w_list=[w1,w2,w3]
    return w_list

def corput(n, b=3):
    q=0
    bk=1/b
    while n>0:
        n, rem= np.divmod(n,b)
        q += rem*bk
        bk /= b
    return q

def halton(n,d):
    x=np.zeros(d)
    base=np.array([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,233,239,241,251,257,263,269,271,277,281])
    assert base.shape[0]>d, "Error: d exceeds the number of basis elements."
    for i in range(d):
        x[i]=corput(n,base[i])
    return x

def A_generate(bij,w_list):
    if type(bij)==list or type(bij)==np.ndarray:
        dim=len(w_list)
        mat_list=[np.outer(w_list[i],w_list[i]) for i in range(dim)]
        for i in range(dim):
            mat_list[i]=mat_list[i]/(bij[i]**2)
        A=sum(mat_list)
        return A
    else:
        dim=len(w_list)
        mat_list=[np.outer(w_list[i],w_list[i]) for i in range(dim)]
        for i in range(dim):
            mat_list[i]=mat_list[i]/(bij**2)
        A=sum(mat_list)
        return A

def w_gen_2pion(m1,m2): #We want to control individual masses being input to our w generator.
    w1=np.zeros((2,1))
    w2=np.zeros((2,1))
    w3=np.zeros((2,1))
    w1[0,0]=m1/(m1+m2)-1
    w2[0,0]=-m1/(m1+m2) + 1
    w2[1,0]=1
    w3[0,0]=-m1/(m1+m2)
    w3[1,0]=1
    wlist=[w1,w2,w3]
    return wlist

def w_gen_2pionCoM(m1,m2,m3):
    w1=np.zeros((2,1))
    w2=np.zeros((2,1))
    w3=np.zeros((2,1))
    w1[0,0]=-m1/(m1+m2)+1
    w1[1,0]=1-(m1+m2)/(m1+m2+m3)
    w2[0,0]=-m1/(m1+m2) 
    w2[1,0]=1-(m1+m2)/(m1+m2+m3)
    w3[0,0]=0
    w3[1,0]=-(m1+m2)/(m1+m2+m3)
    wlist=[w1,w2,w3]
    return wlist