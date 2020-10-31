from sympy import *

import numpy as np
import math

class XOR_SVM:
    def __init__(self):
        pass
    
    def generate_data(self):
        X=np.mat([[1,1],[-1,-1],[1,-1],[-1,1]])
        y=[-1,-1,1,1]
        return X,y
    
    def quadratic_kernel(self, X, y):
        return (1+X*y.T)**2

    def update_loss(self, X, y, step, alpha):
        m=4;
        t=var('a:4')
        dim_1=0
        dim_2=0
        dim_3=0
        dim_4=0
        dim_5=0
        dim_6=0
        for i in range(1,m):
            dim_1-=t[i]*(y[i]*y[0])
            dim_2+=t[i]
            dim_3+=t[i]*y[i]*self.quadratic_kernel(X[i],X[0])
            for j in range(1,m):
                dim_4+=t[i]*y[i]*self.quadratic_kernel(X[i],X[j])*y[j]*t[j]
        first_part=dim_1+dim_2
        second_part=(-1/2)*(pow(dim_1,2)*self.quadratic_kernel(X[0],X[0])+2*y[0]*dim_1*dim_3+dim_4)
        for i in range(1,m):
            dim_5+=log(t[i])
        dim_6=log(dim_1)
        loss=first_part+second_part+1/(step**2)*(dim_5+dim_6)
        #return loss
        return loss.subs({a1:alpha[0],a2:alpha[1],a3:alpha[2]})
    
    def update_weight(self, X, y, k, step, alpha):
        m=4
        t=var('a:4')
        dim_1=0
        dim_2=0
        dim_3=0
        dim_4=0
        dim_5=0
        dim_6=0
        for i in range(1,m):
            dim_1-=t[i]*(y[i]*y[0])
            dim_2+=t[i]
            dim_3+=t[i]*y[i]*self.quadratic_kernel(X[i],X[0])
            for j in range(1,m):
                dim_4+=t[i]*y[i]*self.quadratic_kernel(X[i],X[j])*y[j]*t[j]
        first_part=dim_1+dim_2
        second_part=(-1/2)*(pow(dim_1,2)*self.quadratic_kernel(X[0],X[0])+2*y[0]*dim_1*dim_3+dim_4)
        for i in range(1,m):
            dim_5+=log(t[i])
        dim_6=log(dim_1)
        loss=first_part+second_part+(1/step**2)*(dim_5+dim_6)
        diff_loss=diff(loss,t[k])
        return diff_loss.subs({a1:alpha[0],a2:alpha[1],a3:alpha[2]})
    
    def svm_solver(self, X, y):
        m=4;
        alpha=[]
        alpha_1=0
        for i in range(m-1):
            alpha.append(1) 
        print("initial weight vector is: ", alpha)
        beta = 0.01
        tol_L = 0.00001
        steps=1
        L=1
        updated_L=2;
        while (abs(updated_L-L)>tol_L):
            L=self.update_loss(X,y,steps,alpha)
            print("loss updated")
            for i in range(m-1):
                update=self.update_weight(X,y,i+1,steps,alpha)
                print('weight updated')
                alpha[i]=alpha[i]+beta*update
            steps=steps+1
            updated_L=self.update_loss(X,y,steps,alpha)
        for i in range(m-1):
            alpha_1-=alpha[i]*y[i+1]*y[0]
        alpha.append(alpha_1);
        print(alpha)
        return alpha


def main():
    a=XOR_SVM()
    X, y=a.generate_data()
    a.svm_solver(X, y)

if __name__=="__main__":
    main()

