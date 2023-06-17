import numpy as np
import warnings as wr

class svector:

    def __init__(self,a):
        self.a=a

    def __str__(self):
        return 'svector: \n'+ str(self.a)

    def create_shift(dim, p=None, d=None):
        if p!=None:
            a=np.zeros((3,dim))
            for i in range(dim):
                a[0,i]=1
            a=svector(a)
            return a
        elif d!=None:
            a=np.zeros((3,dim))
            b=np.zeros((3,dim))
            for i in range(dim):
                a[0,i]=1
                b[1,i]=1
            a=svector(a)
            b=svector(b)
            return a,b  

    def create_pion_shift(dim):
        a=np.zeros((3,dim))
        b=np.zeros((3,dim),dtype=complex)
        for i in range(dim):
            a[2,i]=1
            b[0,i]=1
            b[1,i]=1j
            bH=b.conj()
        a=svector(a)
        b=svector(b)
        bH=svector(bH)
        return a,b, bH
#        elif d!=None:
#            a=np.zeros((3,dim))
#            b=np.zeros((3,dim))
#            for i in range(dim):
#                a[0,i]=1
#                b[1,i]=1
#            a=svector(a)
#            b=svector(b)
#            return a,b     

    def PionShift2():
        a1,a2,b1=np.zeros((2,1),dtype=object),np.zeros((2,1),dtype=object),np.zeros((2,1),dtype=object)
        b2,bH1,bH2=np.zeros((2,1),dtype=object),np.zeros((2,1),dtype=object),np.zeros((2,1),dtype=object)
        a2minus=np.zeros((2,1),dtype=object)
        fill=np.zeros((1,3))
        z=np.zeros((1,3))
        zminus=np.zeros((1,3))
        b=np.zeros((1,3),dtype=complex)
        bH=np.zeros((1,3),dtype=complex)
        z[0,2]=1
        zminus[0,2]=-1
        b[0,0]=1
        b[0,1]=1j
        bH[0,0]=1
        bH[0,1]=-1j
        z=svector(z)
        zminus=svector(zminus)
        b=svector(b)
        bH=svector(bH)
        fill=svector(fill)
        a1[0,0]=z
        a1[1,0]=fill
        a2[1,0]=z
        a2[0,0]=fill
        a2minus[1,0]=zminus
        a2minus[0,0]=fill
        b1[0,0]=b
        b1[1,0]=fill
        b2[1,0]=b
        b2[0,0]=fill
        bH1[0,0]=bH
        bH1[1,0]=fill
        bH2[1,0]=bH
        bH2[0,0]=fill
        return a1,a2,a2minus,b1,b2,bH1,bH2

    def __mul__(self,other):
        if isinstance(other,self.__class__):
            dot=0
            for i in range(self.a.shape[1]):
                dot+=self.a[:,i].T@other.a[:,i]
                if np.imag(dot)!=0:
                    wr.warn("ComplexWarning: Dot product contains complex values that will be ommitted in final result") 
                dot=np.real(dot)
            return dot
        elif isinstance(other,np.ndarray): 
            prod=np.zeros((self.a.shape))
            prod_s_r=np.zeros((self.a.shape))
            prod_s_c=np.zeros((self.a.shape))
            for i in range(self.a.shape[1]):
                for j in range(self.a.shape[1]):
                    prod=prod.astype(complex) #Prepared to treat complex numbers
                    r_a=np.real(self.a[:,i])
                    c_a=np.imag(self.a[:,i]) #Splits a into real and complex part
                    prod_r=r_a*other[i,j]
                    prod_c=c_a*other[i,j] #Doint product on individual parts
                    prod[:,j]=prod_r + prod_c*1j #Collected as one single vector
                prod_s_r[:,i]=np.sum(np.real(prod),axis=1)
                prod_s_c[:,i]=np.sum(np.imag(prod),axis=1) #Same as before when handling complex numbers
                prod_s=prod_s_r + prod_s_c*1j
            prod_s=svector(prod_s) #Elements will always have type "complex", but this is fine as long as they are dotted on another svector.
            return prod_s
        elif isinstance(other, int):
            return svector(other*self.a)
        elif isinstance(other,float):
            return svector(other*self.a)
        elif isinstance(other,complex):
            return svector(other*self.a)
        else:
            raise TypeError("unsupported operand type(s) for *: '{}' and '{}'").format(self.__class__, type(other))

    def __add__(self,other):
        if isinstance(other,svector):
            return svector(other.a+self.a)
        elif isinstance(other,float):
            return svector(self.a+other)
        elif isinstance(other,int):
            return svector(self.a+other)


    def __rmul__(self,other):
        if isinstance(other, int):
            return other*self.a
        elif isinstance(other,float):
            return other*self.a
        #else:
        #    raise TypeError("unsupported operand type(s) for *: '{}' and '{}'").format(self.__class__, type(other))
