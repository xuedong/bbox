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

#########
# Utils #
#########

def std_box(f, fmax, nsplits, dim, side):
    box = target.Box(f, fmax, nsplits, dim, side)
    box.std_noise(SIGMA)
    box.std_partition()

    return box

# Regret for HOO
def regret_hoo(bbox, rho, nu, alpha):
    y = np.zeros(HORIZON)
    #y = [0. for i in range(HORIZON)]
    htree = hoo.HTree(bbox.support, None, 0, rho, nu, bbox)
    cum = 0.

    for i in range(HORIZON):
        if UPDATE and alpha < math.log(i+1) * (SIGMA ** 2):
            alpha += 1
            htree.update(alpha)
        x, _, _ = htree.sample(alpha)
        cum += bbox.fmax - bbox.f_mean(x)
        y[i] = cum/(i+1)

    return y

# Plot regret curve
def show(data, data_poo, epoch, horizon, rhos_hoo, rhos_poo, delta):
    length_hoo = len(rhos_hoo)
    length_poo = len(rhos_poo)
    rhostoshow = [int(length_hoo*k/4.) for k in range(4)]
    #rhostoshow = [0, 6, 12, 18]
    #rhostoshow = [0, 1, 3, 7, 15]
    style = [[5,5], [1,3], [5,3,1,3], [5,2,5,2,5,10]]
    #style = [[5,5], [1,3], [5,3,1,3], [5,2,5,2,5,10], [3, 1]]

    means = [[sum([data[k][j][i] for k in range(epoch)])/float(epoch) for i in range(horizon)] for j in range(length_hoo)]
    devs = [[math.sqrt(sum([(data[k][j][i]-means[j][i])**2 for k in range(epoch)])/(float(epoch)*float(epoch-1))) for i in range(horizon)] for j in range(length_hoo)]

    means_poo = [sum([data_poo[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]

    X = np.array(range(horizon))
    for i in range(len(rhostoshow)):
        k = rhostoshow[i]
        label__ = r"$\mathtt{HOO}, \rho = " + str(float(k)/float(length_hoo)) + "$"
        pl.plot(X, np.array(means[k]), label=label__, dashes=style[i])
    pl.plot(X, np.array(means_poo), label=r"$\mathtt{POO}$")
    pl.legend()
    pl.xlabel("number of evaluations")
    pl.ylabel("simple regret")
    pl.show()

    X = np.array(map(math.log, range(horizon)[1:]))
    for i in range(len(rhostoshow)):
        k = rhostoshow[i]
        label__ = r"$\mathtt{HOO}, \rho = " + str(float(k)/float(length_hoo)) + "$"
        pl.plot(X, np.array(map(math.log, means[k][1:])), label=label__, dashes=style[i])
    pl.plot(X, np.array(map(math.log, means_poo[1:])), label=r"$\mathtt{POO}$")
    pl.legend(loc=3)
    pl.xlabel("number of evaluations (log-scale)")
    pl.ylabel("simple regret")
    pl.show()

    X = np.array([float(j)/float(length_hoo-1) for j in range(length_hoo)])
    Y = np.array([means[j][horizon-1] for j in range(length_hoo)])
    Z1 = np.array([means[j][horizon-1]+math.sqrt(2*(devs[j][horizon-1] ** 2) * math.log(1/delta)) for j in range(length_hoo)])
    #Z1 = np.array([means[j][horizon-1]+2*devs[j][horizon-1] for j in range(length_hoo)])
    Z2 = np.array([means[j][horizon-1]-math.sqrt(2*(devs[j][horizon-1] ** 2) * math.log(1/delta)) for j in range(length_hoo)])
    #Z2 = np.array([means[j][horizon-1]-2*devs[j][horizon-1] for j in range(length_hoo)])
    pl.plot(X, Y)
    pl.plot(X, Z1, color="green")
    pl.plot(X, Z2, color="green")
    pl.xlabel(r"$\rho$")
    pl.ylabel("simple regret after " + str(HORIZON) + " evaluations")
    pl.show()

    X = np.array([float(j)/float(length-1) for j in range(length_hoo)])
    Y = np.array([means[j][horizon-1] for j in range(length_hoo)])
    E = np.array([math.sqrt(2*(devs[j][horizon-1] ** 2) * math.log(1/delta)) for j in range(length_hoo)])
    #E = np.array([2*devs[j][horizon-1] for j in range(length_hoo)])
    pl.errorbar(X, Y, yerr=E, color='black', errorevery=3)
    pl.xlabel(r"$\rho$")
    pl.ylabel("simple regret after " + str(HORIZON) + " evaluations")
    pl.show()

# How to choose rhos to be used
def get_rhos(nsplits, rhomax, horizon):
    dmax = math.log(nsplits)/math.log(1./rhomax)
    n = 0
    N = 1
    rhos = np.array([rhomax])

    while n < horizon:
        if n == 0 or n == 1:
            threshold = -float("inf")
        else:
            threshold = 0.5 * dmax * math.log(n/math.log(n))

        while N <= threshold:
            for i in range(N):
                rho_current = math.pow(rhomax, 2.*N/(2*(i+1)))
                rhos = np.append(rhos, rho_current)
            n = 2*n
            N = 2*N

        n = n+N

    return np.unique(np.sort(rhos))

#########################
# Tests for HOO methods #
#########################

start_time = time.time()

# First test
f1 = target.DoubleSine(0.3, 0.8, 0.5)
bbox1 = std_box(f1.f, f1.fmax, NSPLITS, 1, (0., 1.))

f2 = target.DiffFunc(0.5)
bbox2 = std_box(f2.f, f2.fmax, NSPLITS, 1, (0., 1.))

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
bbox3 = std_box(f3.f, f3.fmax, NSPLITS, 2, (-6., 6.))
#bbox3.plot2D()

# Computing regrets
pool = mp.ProcessingPool(JOBS)
def partial_regret_hoo(rho):
    return regret_hoo(bbox2, rho, nu_, alpha_)
#partial_regret_hoo = ft.partial(regret_hoo, bbox=bbox1, nu=nu_, alpha=alpha_)

data = [None for k in range(EPOCH)]
current = [[0. for i in range(HORIZON)] for j in range(RHOMAX)]
rhos_poo = get_rhos(NSPLITS, 0.9, HORIZON)

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
show(data, dataPOO, EPOCH, HORIZON, rhos_hoo, rhos_poo, DELTA)

########################
# Tests for BO methods #
########################
