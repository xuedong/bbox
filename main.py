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
SIDE = (0, 2*np.pi)
DIM = 1
SIGMA = 0.01
HORIZON = 30

VERBOSE = True
PLOT = True
SAVE = True

#target = target.Gramacy1()
target = target.Sine2()
bbox = utils_oo.std_box(target.f, target.fmax, NSPLITS, DIM, SIDE, SIGMA)

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

    xbest, model, info = solve_bayesopt(f, bounds, solver='lbfgs', niter=HORIZON, verbose=VERBOSE)

    mu, s2 = model.predict(x[:, None])

    if PLOT:
        ax = figure().gca()
        ax.plot_banded(x, mu, 2*np.sqrt(s2))
        ax.axvline(xbest)
        ax.scatter(info.x.ravel(), info.y)
        ax.figure.canvas.draw()
        show()

if __name__ == '__main__':
    main()
