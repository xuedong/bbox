#!/usr/bin/env python

# This code is based on the original code of Emile Contal in Matlab
# econtal.person.math.cnrs.fr/software

import numpy as np
import math
import sys

def basis_none(X):
    return np.array([])

def basis_cst(X):
    m, n = X.shape
    return np.ones(m)

def sq_dist(A, B):
    d1, n = A.shape
    d2, m = B.shape

    mu = float(m)/float(n+m)*np.mean(B, 1) + float(n)/float(n+m)*np.mean(A, 1)
    #print(mu)
    A = A - np.reshape(np.tile(mu, (1, n)), A.shape)
    B = B - np.reshape(np.tile(mu, (1, m)), B.shape)

    C = np.reshape(np.tile(sum(np.multiply(A, A), 0), (m, 1)), (m, n)).T \
            + np.reshape(np.tile(sum(np.multiply(B, B), 0), (n, 1)), (n, m)) \
            - 2*np.dot(A.T, B)

    return C
