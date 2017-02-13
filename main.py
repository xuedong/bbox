#!/usr/bin/env python

import pylab as pl
import numpy as np
import os
import sys
import math
import random
import time

import target
import hoo

# Macros
HORIZON = 5000
RHOMAX = 20
VAR = 0.1

# Parameters
rho_ = np.arange(0, 1, 0.01)
nu_ = 1.

#########
# Utils #
#########

def std_box(f, fmax):
    box = target.Box(f, fmax)
    box.std_noise(VAR)
    box.std_partition()

    return box

# Regret for HOO
def regret_hoo(bbox, rho, nu):
    y = np.zeros(HORIZON)
    #y = [0. for i in range(HORIZON)]
    htree = hoo.HTree(bbox.support, None, 0, rho, nu, bbox)
    cum = 0.

    for i in range(HORIZON):
        alpha = math.log(i+1)
        htree.update_node(alpha)
        x = htree.sample(alpha)
        cum += bbox.fmax - bbox.f_mean(x)
        y[i] = cum/(i+1)

    return y

#########
# Tests #
#########

start_time = time.time()

# First test
#f1 = target.DoubleSine(0.3, 0.8, 0.5)
#bbox1 = std_box(f1.f, f1.fmax)
#bbox1.plot()

f2 = target.DiffFunc(0.5)
bbox2 = std_box(f2.f, f2.fmax)
#bbox2.plot()
regrets = np.array([regret_hoo(bbox2, rho_[i], nu_)[HORIZON-1] for i in range(100)])
pl.plot(rho_, regrets)
pl.show()

print("--- %s seconds ---" % (time.time()-start_time))
