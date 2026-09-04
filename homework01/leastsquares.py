# file: leastsquares.py
# 2D point cloud fitting with a line
# run with: $ python leastsquares.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time

def newline(p1, p2, color):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax],color=color)
    ax.add_line(l)
    return l

n = 123 ; x = np.linspace(0.0, 1.0, n)
y_line = -2*x + 3 # original line
# generate a cloud of points randomly away from the original line:
y = y_line + np.random.normal(0, 0.55, n)
A = np.array([x, np.ones(n)])
#print ('A=',A)
A = A.transpose()
#print ('A.transpose()=',A)
# Solve a least squares problem to find best constants to define a fitting line:
start = time.perf_counter()
result = np.linalg.lstsq(A, y, rcond=None)
elapsed = time.perf_counter() - start
print("Wall time:", elapsed)
#print ('result=',result)
a, b = result[0]
p=[(x[i],y[i]) for i in range(len(x))]
p0 = (0,a*0 + b); p1 = (1,a*1 + b)

plt.figure(1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Blue - original line; red - fitted line: ')
plt.legend(['Legend'])
plt.plot(x, y, 'r.')
newline(p0,p1,'red')
newline((0,3),(1,1),'blue')
plt.savefig('leastsquares.png')
plt.show()
