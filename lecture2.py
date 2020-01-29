# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:08:24 2020

@author: noahb

This script is an example for RADI 6022.
This is a multi-line comment.
"""


import numpy as np

#create two arrays with shapes (10, 100, 25) and (10, 15) and entries from uniform distribution [0,1]
A = np.random.random((10,100,15))
B = np.random.random((10,15))

#compute sum over i and sum over k for A_i,j,k*B_i,k
C = np.einsum('ijk,ik->j', A, B)

print(C.shape)


#Example 1: calculate n!
n = 10
output = 1.

for i in range(n):
    output *= (i + 1)
    
print(output) #3628800.0


import time

tic = time.time()
b = np.zeros((int(1e6)))
for a in range(int(1e6)):
    b[a] = np.cos(a)
toc = time.time()
print('Loop time: ' + str(toc - tic))

tic = time.time()
a = np.arange(1e6)
np.cos(a)
toc = time.time()
print('Vector time: ' + str(toc - tic))

import numpy as np
from scipy.special import erfinv

def gaussrand(v, sigma=1.):
    x = np.random.random(v)
    x = (x-0.5)*erfinv(x*np.sqrt(2/np.pi))
    x *= sigma
    return x

rand = gaussrand((3,2))
print(rand)

def my_function():
	print('This is a function from a submodule.')
	return

if __name__ == "__main__":
    print('This code will not be executed on import.')


def print_X():
    X = 'Local variables rule!'
    print(X)
    return

X = 'Local variables drool!'
print_X()
print(X)

def print_args(*args):
    for arg in args:
        print(arg)
        
print_args('a', 'b', 'c')

def average_khan_final(**kwargs):
    total = 0
    for key, value in kwargs.items():
        total += value
    total /= len(kwargs)
    return total














