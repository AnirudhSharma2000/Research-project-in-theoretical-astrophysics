# Research-project-in-theoretical-astrophysics
import matplotlib.pyplot as plt
import numpy as np
import random as R
import math as m
import sympy as sym
import statsmodels.api as sm
from sympy import Symbol

###########################################################################################################
# random number generator
###########################################################################################################
def ran(x,y,z):
    r=R.randrange(x,y,z)
    return r

###########################################################################################################
# function for haar waveleets
###########################################################################################################
def haar(t,k,n):
    hn=[]
    T=(2**n)*t-k
    for i in range(len(T)):
        if 0<=T[i]<1/2:
            hn.append((2**(n/2)))
        elif 1/2<=T[i]<1:
            hn.append((2**(n/2))*(-1))
        else:
            hn.append(0)
    return hn

###########################################################################################################
# function for haar waveleets' integral
###########################################################################################################
def hint(t,k,n):
    hn=[]
    T=(2**n)*t-k
    a=0
    b=1/2
    c=1
    for i in range(len(T)):
        if a<=T[i]<b:
            hn.append((2**((n)/2))*(T[i]-a))
        elif b<=T[i]<=c:
            hn.append(((2**((n)/2))*(2*b-T[i])))
        else:
            hn.append(0)
    return hn

###########################################################################################################
# function generator
###########################################################################################################
x=Symbol('x')
# derivative
z1=[]
def diff(z):
    for i in range(len(z)):
        z1.append(sym.diff(z[i]))
    return np.array(z1)
#function containing possible operating functions with value assigned
def funcs(v,X):
    if v==0:
        z=sym.sin(X)
    elif v==1:
        z=sym.cos(X)
    elif v==2:
        z=X
    elif v==3:
        z=X**2
    elif v==4:
        z=X**3
    elif v==5:
        z=X**4
    elif v==6:
        z=X**5
    elif v==7:
        z=sym.exp(X)
    return z
#function for operations
def operator(z):
    a=[z[0]]
    for i in range(len(z)-1):
        v=ran(0,3,1)
        if v==0:
            a[0]=z[i+1]+a[0]
        elif v==1:
            a[0]=a[0]-z[i+1]
        elif v==2:
            a[0]=a[0]*z[i+1]
    return a

n=4
#loop for applying operators on functions
l=[]
lp=[]
temp=[]
for i in range(n):
    f=ran(0,7,1)
    t=ran(0,2,1)
    f1=funcs(f,x)
    l.append(f1)
temp.append(operator(l))
#lf=temp[0]
#lf=np.array(lf)
lf=[sym.sin(sym.cos(sym.tan(1.52*x)))]
#lp=sym.diff(lf[0],x)
l=np.array(l)
print(lf)

###########################################################################################################
#basis generator
###########################################################################################################
m=4
l0=[]
l1=[]
for i in range(m):
    f0=ran(1,7,1)
    l1.append(funcs(f0,x))
    
#l1p.append(diff(l1))
#bas=np.array(l1)
#print('basis :',bas)

bas = [x**i for i in range(71)]

######################
t=np.arange(0,1,0.001)
######################

############# haar basis #################
n1=8
for i in range(n1):
    for j in range(2**i):
        l0.append(haar(t,j,i))
hl0=np.array(l0)
#print(hl0.shape)
    
######## haar integral basis ############
li=[]
ni=8
for i in range(ni):
    for j in range(2**i):
        li.append(hint(t,j,i))
hi0=np.array(li)
#print(hi0)

###########################################################################################################
#adding value of X
###########################################################################################################
a=t
######## y values ##########
expr1= lf
y1   = sym.lambdify(x, expr1, "numpy")
y    = y1(a)
y    = np.array(y) # array of y(x) of shape 1*n
####### phi matrix ##########
expr = bas
phi1 = sym.lambdify(x, expr, "numpy")
phi2 = phi1(a)
phi2[0] = np.ones(phi2[1].shape)
phi  = np.array(phi2) # array of phi of shape 4*n
phi  = np.transpose(phi) # phi as required
phisq = np.transpose(phi) @ phi # phi T * (phi)  

###### haar phi matrix ##########
hphi   = np.array(hl0)
ones   = np.ones((len(t),1))
hphi   = np.transpose(hphi)
aha    = np.column_stack((ones,hphi))
hphisq = np.transpose(aha) @ aha
#print(aha)
####### haar integeral phi matrix ##########
hiphi  = np.array(hi0)
onesi   = np.arange(0, 1,0.001)
hiphi   = np.transpose(hiphi)
ahai    = np.column_stack((onesi,hiphi))
hiphisq = np.transpose(hiphi) @ hiphi
############################################################################################################
# calculation
############################################################################################################
fx = y @ phi @ np.linalg.inv(phisq)
### basis * weights ####
afunc=fx @ bas
exp = afunc
af1 = sym.lambdify(x, exp.tolist(), "numpy")
af  = af1(a)
af  = np.array(af)

############## haar basis * weights #############
hx = y @ aha @ np.linalg.inv(hphisq + 1.0e-6*np.eye(hphisq.shape[0]))
ah = hx @ np.transpose(aha)

############# haar integral basis * weights #################
Wi = y @ hiphi @ np.linalg.inv(hiphisq)
hif=Wi @ hi0

##################################################
con=np.linalg.cond(np.linalg.inv(phisq))
# print(con)
############################################################################################################
#plotting
############################################################################################################
num=[]
for i in range(len(fx[0])):
    num.append(i)

ax = plt.subplots(1,1, figsize=(12,7))
plt.plot(a,y[0],label='f(x)')
plt.plot(a,ah[0],label='haar wavelet (8)')
plt.plot(a,hif[0],'--',label='haar integral (8)')
plt.plot(a,af[0],'-.',label='monomial basis (71)')
plt.xlabel('X value ->')
plt.ylabel('W * phi')
plt.title('Approximation plots')
############################################################################################################
# plt.plot(num,abs(fx[0]))
# plt.plot(num,abs(Wi[0]))
# plt.plot(num,abs(hx[0]),label='Weights for haar wavelets')
# plt.plot(t,aha[:,1],label='H(1)',linewidth=4)
# plt.plot(t,ahai[:,1],label='Hi(1)',linewidth=4)




# plt.xlabel('number of basis ->',fontsize=25)
# plt.ylabel('Weights',fontsize=25)
# plt.title('Weights plots',fontsize=25)
plt.legend()
plt.rc('legend', fontsize=25)   
plt.rc('xtick',  labelsize=30)     # fontsize of the tick labels
plt.rc('ytick',  labelsize=30)
plt.grid(True, color='gainsboro', linestyle='-', linewidth=0.5)

#############################################################################################################
