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
    A = A - np.tile(mu, (n, 1)).T
    B = B - np.tile(mu, (m, 1)).T

    #print(np.reshape(np.tile(sum(np.multiply(A, A), 0), (m, 1)), (m, n)).T)
    #print(np.reshape(np.tile(sum(np.multiply(B, B), 0), (n, 1)), (n, m)))
    #print(2*np.dot(A.T, B))

    C = np.reshape(np.tile(sum(np.multiply(A, A), 0), (m, 1)), (m, n)).T \
            + np.reshape(np.tile(sum(np.multiply(B, B), 0), (n, 1)), (n, m)) \
            - 2*np.dot(A.T, B)

    return C

def kernel_se(A, B, var, lengthscales):
    ard = np.diag(1./lengthscales)
    D = sq_dist(np.dot(ard, A), np.dot(ard, B))
    K = var * np.exp(-D/2.)

    return K

def kernel_se_norm(A, B):
    scales = np.ones(A.shape[0])
    var = 1
    K = kernel_se(A, B, var, scales)

    return K

def kernel_matern(A, B, var, lengthscales, nu):
    ard = np.diag(1./lengthscales)
    D = np.sqrt(sq_dist(np.sqrt(nu)*np.dot(ard, A), np.sqrt(nu)*np.dot(ard, B)))

    if nu == 1:
        K = var * np.exp(-D)
    elif nu == 3:
        K = var * np.multiply(1+D, np.exp(-D))
    elif nu == 5:
        K = var * np.multiply(1+np.multiply(D, 1+D/3.), np.exp(-D))
    else:
        raise ValueError('kernel_matern is only available for 1, 3, 5')

    return K
