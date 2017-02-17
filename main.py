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

import target
import hoo

# Macros
HORIZON = 2000
RHOMAX = 20
SIGMA = 0.1
EPOCH = 20
VERBOSE = False

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
        htree.update(alpha_)
        x = htree.sample(alpha_)
        cum += bbox.fmax - bbox.f_mean(x)
        y[i] = cum/(i+1)

    return y

# Plot regret curve
def show(data, epoch, horizon, rhomax, func):
    rhos = [int(rhomax*k/4.) for k in range(4)]
    style = [[5,5], [1,3], [5,3,1,3], [5,2,5,2,5,10]]

    means = [[sum([data[k][j][i] for k in range(epoch)])/float(epoch) for i in range(horizon)] for j in range(rhomax)]
    devs = [[math.sqrt(sum([(data[k][j][i]-means[j][i])**2 for k in range(epoch)])/float(epoch-1)) for i in range(horizon)] for j in range(rhomax)]

    X = np.array(range(horizon))
    for i in range(len(rhos)):
        k = rhos[i]
        label__ = r"$\mathtt{HOO}, \rho = " + str(float(k)/float(rhomax)) + "$"
        pl.plot(X, np.array(means[k]), label=label__, dashes=style[i])
    pl.legend()
    pl.xlabel("number of evaluations")
    pl.ylabel("simple regret")
    pl.show()

    X = np.array(map(math.log, range(horizon)[1:]))
    for i in range(len(rhos)):
        k = rhos[i]
        label__ = r"$\mathtt{HOO}, \rho = " + str(float(k)/float(rhomax)) + "$"
        pl.plot(X, np.array(map(math.log, means[k][1:])), label=label__, dashes=style[i])
    pl.legend(loc=3)
    pl.xlabel("number of evaluations (log-scale)")
    pl.ylabel("simple regret")
    pl.show()

    X = np.array([float(j)/float(rhomax-1) for j in range(rhomax)])
    Y = np.array([means[j][horizon-1] for j in range(rhomax)])
    Z1 = np.array([means[j][horizon-1]+2*devs[j][horizon-1] for j in range(rhomax)])
    Z2 = np.array([means[j][horizon-1]-2*devs[j][horizon-1] for j in range(rhomax)])
    pl.plot(X, Y)
    pl.plot(X, Z1, color="green")
    pl.plot(X, Z2, color="green")
    pl.xlabel(r"$\rho$")
    pl.ylabel("simple regret after 2000 evaluations")
    pl.show()

    X = np.array([float(j)/float(rhomax-1) for j in range(rhomax)])
    Y = np.array([means[j][horizon-1] for j in range(rhomax)])
    E = np.array([2*devs[j][horizon-1] for j in range(rhomax)])
    pl.errorbar(X, Y, yerr=E, errorevery=3)
    pl.xlabel(r"$\rho$")
    pl.ylabel("simple regret after 2000 evaluations")
    pl.show()

#########
# Tests #
#########

start_time = time.time()

# First test
f1 = target.DoubleSine(0.3, 0.8, 0.5)
bbox1 = std_box(f1.f, f1.fmax)
#bbox1.plot()

#f2 = target.DiffFunc(0.5)
#bbox2 = std_box(f2.f, f2.fmax)
#bbox2.plot()

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

data = [None for k in range(EPOCH)]
current = [[0. for i in range(HORIZON)] for j in range(RHOMAX)]
for k in range(EPOCH):
    for j in range(RHOMAX):
        regrets = regret_hoo(bbox1, float(j)/float(RHOMAX), alpha_)
        for i in range(HORIZON):
            current[j][i] += regrets[i]
    data[k] = current
    current = [[0. for i in range(HORIZON)] for j in range(RHOMAX)]

show(data, EPOCH, HORIZON, RHOMAX, f1)
