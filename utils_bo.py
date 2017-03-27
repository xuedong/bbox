#!/usr/bin/env python

# This code is based on the original code of Emile Contal in Matlab
# econtal.person.math.cnrs.fr/software

import numpy as np

def cummax(R):
    d = R.shape[0]
    n = R.shape[1]
    M = np.zeros((d, n))

    maxi = R[:, 0]
    M[:, 0] = maxi

    for i in range(1, n):
        maxi = np.maximum(maxi, R[:, i])
        print(maxi)
        M[:, i] = maxi
        print(M[:, i])

    return M
