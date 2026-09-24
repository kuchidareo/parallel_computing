# the code computes a 2D list of random numbers, performLU-factorisation
#   and solves the system. Residual of the result being checked.
import random
import numpy as np


def solver(a,b): # LU-factorisation
    for k in range(n):
        if a[k, k] == 0.0:
            raise ValueError(f"{k}-th diagonal element is zero")

        a[k + 1:, k] /= a[k, k]

        a[k + 1:, k + 1:] -= np.outer(
            a[k + 1:, k],
            a[k, k + 1:]
        )

    # Forward substitution
    y = np.empty(n)

    for i in range(n):
        y[i] = b[i] - a[i, :i] @ y[:i]

    # Backward substitution
    x = np.empty(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            y[i] - a[i, i + 1:] @ x[i + 1:]
        ) / a[i, i]

    return x

n = 10
# a=[]
# for i in range(n):
#     a.append([])
#     for j in range(n):
#         a[i].append(random.random())
a = np.random.rand(n, n)

# b=[]
# for i in range(n):
#     b.append(random.random())
b = np.random.rand(n)

# copy the original list of lists (in python2 - no .copy() method yet...)
# aa=list(a)
# for i in range(n):
#     aa[i]=list(a[i])
aa = a.copy()

x=solver(a,b)
# check the result - we ensure that matrix-vector application Ax to obtained
# solution indeed gives a vector that is fairly closed to desired vector b:
# z=[]
# me=0.0
# for i in range(n):
#     z.append(0.0)
#     for j in range(n):
#         z[i]=z[i] + aa[i][j]*x[j]
#     e=abs(z[i]-b[i])
#     if e > me:
#         me=e
z = np.matmul(aa, x)
me = np.max(np.abs(z - b))

np.testing.assert_allclose(np.dot(aa,x), b)
print('max error:',me)
