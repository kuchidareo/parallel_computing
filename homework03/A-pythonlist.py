# the code computes a 2D list of random numbers, performLU-factorisation
#   and solves the system. Residual of the result being checked.
import random
import numpy as np


def solver(a,b): # LU-factorisation
    for k in range(n):
        if a[k][k] == 0.0:
            print(k,'-th diagonal element is zero - exiting')
            exit(-1)
        for i in range(k+1,n):
            a[i][k] = a[i][k] / a[k][k]
        for j in range(k+1,n):
            for i in range(k+1,n):
                a[i][j]=a[i][j]-a[i][k]*a[k][j]
    y=[] # forward substitution - solve Ly=b
    y.append(b[0])
    for i in range(1,n):
        y.append(b[i])
        for j in range(i):
            y[i]=y[i]-a[i][j]*y[j]
    x=[] # backward substitution - solve Ux=y
    for i in range(n):
        x.append(0.0)
    x[n-1]=y[n-1]/a[n-1][n-1]
    for i in range(n-2,-1,-1):
        x[i]=y[i]
        for j in range(i+1,n):
            x[i] = x[i]-a[i][j]*x[j]
        x[i] = x[i] / a[i][i]
    return x

n = 10
a=[]
for i in range(n):
    a.append([])
    for j in range(n):
        a[i].append(random.random())

b=[]
for i in range(n):
    b.append(random.random())

# copy the original list of lists (in python2 - no .copy() method yet...)
aa=list(a)
for i in range(n):
    aa[i]=list(a[i])

x=solver(a,b)
# check the result - we ensure that matrix-vector application Ax to obtained
# solution indeed gives a vector that is fairly closed to desired vector b:
z=[]
me=0.0
for i in range(n):
    z.append(0.0)
    for j in range(n):
        z[i]=z[i] + aa[i][j]*x[j]
    e=abs(z[i]-b[i])
    if e > me:
        me=e

print('max error:',me)
