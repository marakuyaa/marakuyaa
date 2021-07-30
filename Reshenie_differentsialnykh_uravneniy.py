#!/usr/bin/env python
# coding: utf-8

# ### Метод Рунге Кутты
# #### Четвёртый порядок сходимости

# In[2]:


import math as mt
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


def runge_kutta(f, u0, t0, tN, h):
    N = int((tN - t0) / h)
    u = np.zeros(N)
    u[0] = u0
    for n in range (0, N-1):
        tn = t0 + n * h
        k1 = h * f(tn , u[n])
        k2 = h * f(tn + h/2, u[n] + k1/2)
        k3 = h * f(tn + h/2, u[n] + k2/2)
        k4 = h * f(tn + h, u[n] + k3)
        u[n+1] = u[n] + 1/6 * (k1 + 2*k2+ 2*k3 +k4)
    return u


# In[4]:


def f(t, u):
    return t
u0 = 0
t = np.linspace(-5, 5, 10000)
u = runge_kutta(f, 0, -5, 5, 0.001)
plt.plot(t, u)


# ### u"+2u'+1=0
# График разности численного и аналитического решений

# In[11]:


x0 = 0
xN = 10
N = 10000
x = np.linspace(x0, xN, N)
u = np.zeros(N)
u[0] = 1
u[1] = 1
h = (xN - x0) / N
for n in range(1, N-1):
    u[n+1] = u[n] * (2-h**2)/(1+h) + u[n-1] * (h-1)/(h+1) #+ h**2/(h+1) * x[n]**2 
plt.figure(figsize=(16,12))
plt.grid(True)
plt.plot(x, u-y)
#plt.plot(x,y)


# #### Аналитическое решение

# In[10]:


x0 = 0
xN = 10
N = 10000
x = np.linspace(x0, xN, N)
def f(x):
    return x*mt.exp(-x) + mt.exp(-x) #+x**2 - 4*x + 6 + 
y = np.zeros(N)
print (f(1))
for n in range (0, N):
    y[n] = f(x[n])
plt.plot(x,y)


# In[ ]:




