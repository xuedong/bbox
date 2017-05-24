#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

import pylab as pl
import numpy as np
import os
import sys
import math
import random
import pickle
import time
import pathos.multiprocessing as mp
import functools as ft

import target
import hoo
import hct
import poo
import utils_oo

import basis
import gp
import utils_bo

# System settings
sys.setrecursionlimit(10000)
#random.seed(42)

# Macros
HORIZON = 5000
RHOMAX = 20
SIGMA = 0.1
DELTA = 0.05
EPOCH = 50
UPDATE = False
VERBOSE = True
PARALLEL = True
JOBS = 8
NSPLITS = 3
SAVE = True
PATH = "data/"
PLOT = False

# Global Parameters
alpha_ = math.log(HORIZON) * (SIGMA ** 2)
rhos_hoo = [float(j)/float(RHOMAX-1) for j in range(RHOMAX)]
nu_ = 1.

start_time = time.time()

# First test
f1 = target.DoubleSine(0.3, 0.8, 0.5)
bbox1 = utils_oo.std_box(f1.f, f1.fmax, NSPLITS, 1, (0., 1.), SIGMA)
#bbox1.plot1D()

f2 = target.DiffFunc(0.5)
bbox2 = utils_oo.std_box(f2.f, f2.fmax, NSPLITS, 1, (0., 1.), SIGMA)
#bbox2.plot1D()

f5 = target.Sine1()
bbox5 = utils_oo.std_box(f5.f, f5.fmax, NSPLITS, 1, (0., np.pi), SIGMA)
#bbox5.plot1D()

f6 = target.Gramacy1()
bbox6 = utils_oo.std_box(f6.f, f6.fmax, NSPLITS, 1, (0.5, 2.5), SIGMA)
#bbox6.plot1D()

f7 = target.Sine2()
bbox7 = utils_oo.std_box(f7.f, f7.fmax, NSPLITS, 1, (0., 2*np.pi), SIGMA)

# Simple regret evolutiion with respect to different rhos
#regrets = np.zeros(RHOMAX)
#for k in range(EPOCH):
#    regrets_current = np.array([regret_hoo(bbox2, rho_[i], nu_)[HORIZON-1] for i in range(RHOMAX)])
#    regrets = np.add(regrets, regrets_current)
#print("--- %s seconds ---" % (time.time() - start_time))
#pl.plot(rho_, regrets/float(EPOCH))
#pl.show()

# Simple regret evolution with respect to number of evaluations
regrets1 = np.zeros(HORIZON)
for k in range(EPOCH):
    regrets_current, _, _ = utils_oo.regret_hoo(bbox6, 0.66, nu_, alpha_, SIGMA, HORIZON, UPDATE)
    regrets1 = np.add(regrets1, regrets_current)
#print("--- %s seconds ---" % (time.time() - start_time))
X = range(HORIZON)
pl.plot(X, regrets1/float(EPOCH), label=r"$\mathtt{HOO}$", color='red')
#pl.show()

regrets2 = np.zeros(HORIZON)
c_ = 2 * math.sqrt(1/(1-0.66))
c1_ = (0.66/(3*nu_)) ** (1./8)
for k in range(EPOCH):
    regrets_current, _, _ = utils_oo.regret_hct(bbox6, 0.66, nu_, c_, c1_, DELTA, SIGMA, HORIZON)
    regrets2 = np.add(regrets2, regrets_current)
print("--- %s seconds ---" % (time.time() - start_time))
#X = range(HORIZON)
pl.plot(X, regrets2/float(EPOCH), label=r"$\mathtt{HCT}$", color='blue')
pl.show()

# 2D function test
f3 = target.Himmelblau()
bbox3 = utils_oo.std_box(f3.f, f3.fmax, NSPLITS, 2, (-6., 6.), SIGMA)
#bbox3.plot2D()

f4 = target.Rosenbrock(1, 100)
bbox4 = utils_oo.std_box(f4.f, f4.fmax, NSPLITS, 2, (-3., 3.), SIGMA)
#bbox4.plot2D()

########################
# Tests for OO methods #
########################
"""
# Computing regrets
pool = mp.ProcessingPool(JOBS)
def partial_regret_hoo(rho):
    cum, sim, sel = utils_oo.regret_hoo(bbox6, rho, nu_, alpha_, SIGMA, HORIZON, UPDATE)
    return cum
#partial_regret_hoo = ft.partial(utils_oo.regret_cumulative_hoo, bbox=bbox1, nu=nu_, alpha=alpha_, SIGMA, HORIZON, UPDATE)

data = [None for k in range(EPOCH)]
current = [[0. for i in range(HORIZON)] for j in range(RHOMAX)]
rhos_poo = utils_oo.get_rhos(NSPLITS, 0.9, HORIZON)

# HOO
if VERBOSE:
    print("HOO!")

if PARALLEL:
    for k in range(EPOCH):
        data[k] = pool.map(partial_regret_hoo, rhos_hoo)
        if VERBOSE:
            print(str(k+1)+"/"+str(EPOCH))
        #print(regrets.shape)
else:
    for k in range(EPOCH):
        for j in range(len(rhos_hoo)):
            cums, sims, sels = utils.regret_hoo(bbox6, float(j)/float(len(rhos_hoo)), nu_, alpha_, SIGMA, HORIZON, UPDATE)
            for i in range(HORIZON):
                current[j][i] += cums[i]
            if VERBOSE:
                print(str(1+j+k*len(rhos_hoo))+"/"+str(EPOCH*len(rhos_hoo)))
        data[k] = current
        current = [[0. for i in range(HORIZON)] for j in range(len(rhos_hoo))]

#print(data[0])

# POO
if VERBOSE:
    print("POO!")

pcum, psim, psel = utils_oo.regret_poo(bbox6, rhos_poo, nu_, alpha_, HORIZON, EPOCH)
data_poo = pcum

#print(dataPOO)
print("--- %s seconds ---" % (time.time() - start_time))

if SAVE and VERBOSE:
    print("Writing data...")
    for k in range(EPOCH):
        with open("data/HOO"+str(k+1), 'wb') as file:
            pickle.dump(data[k], file)
        print(str(k+1)+"/"+str(EPOCH))
    with open("data/POO", 'wb') as file:
        pickle.dump(data_poo, file)

utils_oo.show(PATH, EPOCH, HORIZON, rhos_hoo, rhos_poo, DELTA)
"""
########################
# Tests for BO methods #
########################
"""
#f, Xs, Ys, Xt, Yt, Kss = utils_bo.sample(side=(0, 40), plot=True, bbox=None)

#utils_bo.bo(f, Xt, Yt, Xs, 100, bbox=False)

a, b = (0., 2*np.pi)
bounds = np.array([[a, b]])
utils_bo.animated(f7, bounds, True, VERBOSE, HORIZON)
"""
