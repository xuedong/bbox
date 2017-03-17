#!/usr/bin/env python

# This code is based on the original code of Emile Contal in Matlab
# econtal.person.math.cnrs.fr/software

import numpy as np
import basis

def solve_chol(R, Y):
    return np.linalg.solve(R, np.linalg.solve(R.T, Y))
