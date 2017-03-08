#!/usr/bin/env python

# This code is based on the original code of Emile Contal in Matlab
# econtal.person.math.cnrs.fr/software

import numpy as np
import math
import sys

def gpucb(var_post, log_prob, nb_tests):
    return math.sqrt(2.*(log_prob+math.log(nb_tests))*var_post)

def basis_none(X):
    return np.array([])

def basis_cst(X):
    m, n = X.shape
    return np.ones(m)

def sq_dist(A, B):
    d1, n = A.shape
    d2, m = B.shape

    mu = float(m)/float(n+m)*np.mean(B, 1) + float(n)/float(n+m)*np.mean(A, 1)
    print(mu)
    print(np.tile(A, (1, n)))
    A = A - np.tile(A, (1, n))
    B = B - np.tile(B, (1, m))

    return mu
