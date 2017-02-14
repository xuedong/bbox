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
HORIZON = 2000
RHOMAX = 100
SIGMA = 0.1
EPOCH = 64

# Global Parameters
alpha_ = math.log(HORIZON) * (SIGMA ** 2)
rho_ = np.arange(0, 1, 0.01)
nu_ = 1.

#########
# Utils #
#########

def std_box(f, fmax):
    box = target.Box(f, fmax)
    box.std_noise(SIGMA)
    box.std_partition()

    return box

# Regret for HOO
def regret_hoo(bbox, rho, nu):
    y = np.zeros(HORIZON)
    #y = [0. for i in range(HORIZON)]
    htree = hoo.HTree(bbox.support, None, 0, rho, nu, bbox)
    cum = 0.

    for i in range(HORIZON):
        htree.update_node(alpha_)
        x = htree.sample(alpha_)
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

# Simple regret evolutiion with respect to different rhos
regrets = np.zeros(RHOMAX)
for k in range(EPOCH):
    regrets_current = np.array([regret_hoo(bbox2, rho_[i], nu_)[HORIZON-1] for i in range(RHOMAX)])
    regrets = np.add(regrets, regrets_current)
print("--- %s seconds ---" % (time.time() - start_time))
pl.plot(rho_, regrets/float(EPOCH))
pl.show()

# Simple regret evolution with respect to number of evaluations
#regrets = np.zeros(HORIZON)
#for k in range(EPOCH):
#    regrets_current = np.array(regret_hoo(bbox2, 0.66, nu_))
#    regrets = np.add(regrets, regrets_current)
#print("--- %s seconds ---" % (time.time() - start_time))
#X = range(HORIZON)
#pl.plot(X, regrets/float(EPOCH))
#pl.show()
