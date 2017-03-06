#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

import pylab as pl
import numpy as np
import os
import sys
import math
import random
import time
import pathos.multiprocessing as mp
import functools as ft

import target
import hoo
import poo
import utils

# System settings
sys.setrecursionlimit(10000)
#random.seed(42)

# Macros
HORIZON = 5000
RHOMAX = 20
SIGMA = 0.1
DELTA = 0.05
EPOCH = 20
UPDATE = False
VERBOSE = True
PARALLEL = True
JOBS = 8
NSPLITS = 2

# Global Parameters
alpha_ = math.log(HORIZON) * (SIGMA ** 2)
rhos_hoo = [float(j)/float(RHOMAX-1) for j in range(RHOMAX)]
nu_ = 1.

#########################
# Tests for HOO methods #
#########################

start_time = time.time()

# First test
f1 = target.DoubleSine(0.3, 0.8, 0.5)
bbox1 = utils.std_box(f1.f, f1.fmax, NSPLITS, 1, (0., 1.), SIGMA)

f2 = target.DiffFunc(0.5)
bbox2 = utils.std_box(f2.f, f2.fmax, NSPLITS, 1, (0., 1.), SIGMA)

# Simple regret evolutiion with respect to different rhos
#regrets = np.zeros(RHOMAX)
#for k in range(EPOCH):
#    regrets_current = np.array([regret_hoo(bbox2, rho_[i], nu_)[HORIZON-1] for i in range(RHOMAX)])
#    regrets = np.add(regrets, regrets_current)
#print("--- %s seconds ---" % (time.time() - start_time))
#pl.plot(rho_, regrets/float(EPOCH))
#pl.show()

# Simple regret evolution with respect to number of evaluations
#regrets = np.zeros(HORIZON)
#for k in range(EPOCH):
#    regrets_current = np.array(regret_hoo(bbox2, 0.66, nu_))
#    regrets = np.add(regrets, regrets_current)
#print("--- %s seconds ---" % (time.time() - start_time))
#X = range(HORIZON)
#pl.plot(X, regrets/float(EPOCH))
#pl.show()

# 2D function test
f3 = target.Himmelblau()
bbox3 = utils.std_box(f3.f, f3.fmax, NSPLITS, 2, (-6., 6.), SIGMA)
#bbox3.plot2D()

# Computing regrets
pool = mp.ProcessingPool(JOBS)
def partial_regret_hoo(rho):
    return utils.regret_hoo(bbox2, rho, nu_, alpha_, SIGMA, HORIZON, UPDATE)
#partial_regret_hoo = ft.partial(regret_hoo, bbox=bbox1, nu=nu_, alpha=alpha_)

data = [None for k in range(EPOCH)]
current = [[0. for i in range(HORIZON)] for j in range(RHOMAX)]
rhos_poo = utils.get_rhos(NSPLITS, 0.9, HORIZON)

# HOO
if VERBOSE:
    print("HOO!")

if PARALLEL:
    for k in range(EPOCH):
        data[k] = np.array(pool.map(partial_regret_hoo, rhos_hoo))
        if VERBOSE:
            print(str(k+1)+"/"+str(EPOCH))
        #print(regrets.shape)
else:
    for k in range(EPOCH):
        for j in range(len(rhos_hoo)):
            regrets = regret_hoo_bis(bbox2, float(j)/float(len(rhos_hoo)), nu_, alpha_)
            for i in range(HORIZON):
                current[j][i] += regrets[i]
            if VERBOSE:
                print(str(1+j+k*len(rhos_hoo))+"/"+str(EPOCH*len(rhos_hoo)))
        data[k] = current
        current = [[0. for i in range(HORIZON)] for j in range(len(rhos_hoo))]

# POO
if VERBOSE:
    print("POO!")

dataPOO = [[0. for j in range(HORIZON)] for i in range(EPOCH)]
for i in range(EPOCH):
    ptree = poo.PTree(bbox2.support, None, 0, rhos_poo, nu_, bbox2)
    count = 0
    length = len(rhos_poo)
    cum = [0.] * length
    emp = [0.] * length
    smp = [0] * length

    while count < HORIZON:
        for k in range(length):
            x, noisy, existed = ptree.sample(alpha_, k)
            cum[k] += bbox2.fmax - bbox2.f_mean(x)
            count += existed
            emp[k] += noisy
            smp[k] += 1

            if existed and count <= HORIZON:
                best_k = max(range(length), key=lambda x: (-float("inf") if smp[k] == 0 else emp[x]/smp[k]))
                dataPOO[i][count-1] = 1 if smp[best_k] == 0 else cum[best_k]/float(smp[best_k])

print("--- %s seconds ---" % (time.time() - start_time))

#bbox1.plot1D()
utils.show(data, dataPOO, EPOCH, HORIZON, rhos_hoo, rhos_poo, DELTA)

########################
# Tests for BO methods #
########################
