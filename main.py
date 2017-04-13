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
SIDE = (0.5, 2.5)
DIM = 1
SIGMA = 0.01
HORIZON = 10

VERBOSE = True
PLOT = True
SAVE = True

target = target.Gramacy1()
bbox = utils_oo.std_box(target.f, target.fmax, NSPLITS, DIM, SIDE, SIGMA)

def f(x):
    x = np.array([x])
    return target.f(x)

def main():
    a, b = SIDE
    bounds = np.array([[a, b]])
    utils_bo.animated(target, bounds, PLOT, VERBOSE, HORIZON)

if __name__ == '__main__':
    main()
