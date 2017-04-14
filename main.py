#!/usr/bin/env python

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pylab as pl
import os
import sys
import math
import pickle
import time
import pathos.multiprocessing as mp

import hoo
import poo
import target
import utils_oo

import gp
import basis
import acquisition
import utils_bo

from reggie import make_gp, MCMC
from ezplot import figure, show

from pybo import inits
from pybo import policies
from pybo import solvers
from pybo import recommenders
from pybo import solve_bayesopt

NSPLITS = 2
RHOMAX = 20
DELTA = 0.05
JOBS = 8

SIDE = (0.5, 2.5)
DIM = 1
SIGMA = 0.01
HORIZON = 100
EPOCH = 10

VERBOSE = True
PLOT = True
SAVE = True
PARALLEL = True
PATH = "data/"
UPDATE = False

alpha_ = math.log(HORIZON) * (SIGMA ** 2)
rhos_hoo = [float(j)/float(RHOMAX-1) for j in range(RHOMAX)]
nu_ = 1.

target = target.Gramacy1()
#target = target.Sine2()
bbox = utils_oo.std_box(target.f, target.fmax, NSPLITS, DIM, SIDE, SIGMA)

#pool = mp.ProcessingPool(JOBS)

def f(x):
    x = np.array([x])
    return target.f(x)[0]

def main():
    #a, b = SIDE
    #bounds = np.array([[a, b]])
    #utils_bo.animated(target, bounds, PLOT, VERBOSE, HORIZON)

    a, b = SIDE
    bounds = [a, b]
    x = np.linspace(bounds[0], bounds[1], 500)
    regrets = np.zeros(HORIZON)

    if VERBOSE:
        print('GPUCB!')

    infos = [None for k in range(EPOCH)]
    for k in range(EPOCH):
        xbest, model, info = solve_bayesopt(f, bounds, policy='ucb', solver='lbfgs', niter=HORIZON, verbose=False)

        mu, s2 = model.predict(x[:, None])
        infos[k] = info

        #regrets_current = np.array(utils_bo.regret_bo(bbox, info, HORIZON))
        #regrets = np.add(regrets, regrets_current)

        print(k+1)

    #data_ucb = [None for k in range(EPOCH)]
    data_ucb = utils_bo.regret_bo(bbox, infos, HORIZON, EPOCH)

    # Computing regrets
    pool = mp.ProcessingPool(JOBS)
    def partial_regret_hoo(rho):
        cum, sim, sel = utils_oo.regret_hoo(bbox, rho, nu_, alpha_, SIGMA, HORIZON, UPDATE)
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
                cums, sims, sels = utils.regret_hoo(bbox, float(j)/float(len(rhos_hoo)), nu_, alpha_, SIGMA, HORIZON, UPDATE)
                for i in range(HORIZON):
                    current[j][i] += cums[i]
                if VERBOSE:
                    print(str(1+j+k*len(rhos_hoo))+"/"+str(EPOCH*len(rhos_hoo)))
            data[k] = current
            current = [[0. for i in range(HORIZON)] for j in range(len(rhos_hoo))]

    # POO
    if VERBOSE:
        print("POO!")

    pcum, psim, psel = utils_oo.regret_poo(bbox, rhos_poo, nu_, alpha_, HORIZON, EPOCH)
    data_poo = pcum

    #print(dataPOO)
    #print("--- %s seconds ---" % (time.time() - start_time))

    if SAVE and VERBOSE:
        print("Writing data...")
        for k in range(EPOCH):
            with open("data/HOO"+str(k+1), 'wb') as file:
                pickle.dump(data[k], file)
            print(str(k+1)+"/"+str(EPOCH))
        with open("data/POO", 'wb') as file:
            pickle.dump(data_poo, file)
        with open("data/GPUCB", 'wb') as file:
            pickle.dump(data_ucb, file)

    if PLOT:
        utils_oo.show(PATH, EPOCH, HORIZON, rhos_hoo, rhos_poo, DELTA)

    #if PLOT:
    #    X = range(HORIZON)
    #    pl.plot(X, regrets/float(EPOCH))
    #    pl.show()

    #if PLOT:
    #    ax = figure().gca()
    #    ax.plot_banded(x, mu, 2*np.sqrt(s2))
    #    ax.axvline(xbest)
    #    ax.scatter(info.x.ravel(), info.y)
    #    ax.figure.canvas.draw()
    #    show()

if __name__ == '__main__':
    main()
