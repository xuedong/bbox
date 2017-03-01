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
    m, d2 = B.shape
    try:
        d1 == d2
    except:
        e = sys.exc_info()[0]
        write_to_page("<p>Error: column lengths must agree</p>" % e)
