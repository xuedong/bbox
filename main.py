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

NSPLITS = 3
RHOMAX = 20
DELTA = 0.05
JOBS = 8

#SIDE = (.5, 2.5)
SIDE = (0., 2*np.pi)
DIM = 1
SIGMA = 0.01
HORIZON = 50
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

#target = target.Gramacy1()
target = target.Sine2()
#target = target.Himmelblau()
bbox = utils_oo.std_box(target.f, target.fmax, NSPLITS, DIM, SIDE, SIGMA)

#pool = mp.ProcessingPool(JOBS)

def f(x):
    x = np.array([x])
    return target.f(x)[0]

def f2(x):
    a, b = x
    x = np.array([a, b])
    return target.f(x)[0]

def get_info(lsp, f, bounds, policy, solver, recommender, niter, verbose, epoch):
    infos = [None for k in range(epoch)]
    for k in range(epoch):
        xbest, model, info = solve_bayesopt(f, bounds, policy=policy, solver=solver, recommender=recommender, niter=niter, verbose=verbose)

        mu, s2 = model.predict(lsp[:, None])
        infos[k] = info

        print(k+1)

    return infos, mu, s2, xbest, model, info

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

    infos_direct, mu_direct, s2_direct, xbest_direct, model_direct, info_direct = get_info(x, f, bounds, 'ucb', 'direct', 'incumbent', HORIZON, False, EPOCH)

    data_ucb_direct = utils_bo.regret_bo(bbox, infos_direct, HORIZON, EPOCH)

    infos_lbfgs, mu_lbfgs, s2_lbfgs, xbest_lbfgs, model_lbfgs, info_lbfgs = get_info(x, f, bounds, 'ucb', 'lbfgs', 'incumbent', HORIZON, False, EPOCH)

    data_ucb_lbfgs = utils_bo.regret_bo(bbox, infos_lbfgs, HORIZON, EPOCH)

    #infos_ei, mu_ei, s2_ei, xbest_ei, model_ei, info_ei = get_info(x, f, bounds, 'ei', 'direct', 'incumbent', HORIZON, False, EPOCH)

    #data_ei = utils_bo.regret_bo(bbox, infos_ei, HORIZON, EPOCH)

    #data_pi = utils_bo.regret_bo(bbox, infos_pi, HORIZON, EPOCH)

    infos_thomp, mu_thomp, s2_thomp, xbest_thomp, model_thomp, info_thomp = get_info(x, f, bounds, 'thompson', 'direct', 'incumbent', HORIZON, False, EPOCH)

    data_thomp = utils_bo.regret_bo(bbox, infos_thomp, HORIZON, EPOCH)

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
        with open("data/GPUCB_DIRECT", 'wb') as file:
            pickle.dump(data_ucb_direct, file)
        with open("data/GPUCB_LBFGS", 'wb') as file:
            pickle.dump(data_ucb_lbfgs, file)
        #with open("data/EI", 'wb') as file:
        #    pickle.dump(data_ei, file)
        #with open("data/PI", 'wb') as file:
        #    pickle.dump(data_pi, file)
        with open("data/THOMP", 'wb') as file:
            pickle.dump(data_thomp, file)

    if PLOT:
        utils_oo.show(PATH, EPOCH, HORIZON, rhos_hoo, rhos_poo, DELTA)

    if PLOT:
        fig = figure(figsize=(10, 5))
        ax1 = fig.add_subplotspec((1, 2), (0, 0))
        ax2 = fig.add_subplotspec((1, 2), (0, 1))

        ax1.plot_banded(x, mu_direct, 2*np.sqrt(s2_direct))
        ax1.axvline(xbest_direct)
        ax1.scatter(info_direct.x.ravel(), info_direct.y)

        ax2.plot_banded(x, mu_lbfgs, 2*np.sqrt(s2_lbfgs))
        ax2.axvline(xbest_lbfgs)
        ax2.scatter(info_lbfgs.x.ravel(), info_lbfgs.y)

        fig.canvas.draw()

    show()
    bbox.plot1D()

if __name__ == '__main__':
    main()
