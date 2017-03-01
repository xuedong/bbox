import numpy as np
import gpucb

A = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
B = [[1, 2, 3], [1, 2, 3]]
A = np.array(A)
B = np.array(B)

gpucb.sq_dist(A, B)
