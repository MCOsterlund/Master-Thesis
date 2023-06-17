import numpy as np
import gauss_transformer as tr

w_list=tr.w_gen_3()
b=5
x=tr.halton(1,3)
bij=-np.log(x)*b
m_list=np.array([1, 1, np.inf])
K=np.array([[1/2,0,0],[0,1/2,0],[0,0,0]])
J,U=tr.jacobi_transform(m_list)
K_trans=J@K@J.T
w_trans=[U.T @ w_list[i] for i in range(len(w_list))]

dim=len(w_trans)
bij=[]

for i in range(5):
    hal=tr.halton(i+1,len(w_trans))
    b=-np.log(hal)*b
    bij=np.append(bij,b)

for i in range(0,len(bij),dim):
        print(bij[i:i+dim])
        A=tr.A_generate(bij[i:i+dim],w_trans)
