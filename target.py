#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

import numpy as np
import math
import random
#import operator as op
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

import utils

"""
Target functions
"""

class Sine1:
    def __init__(self):
        self.fmax = 0.

    def f(self, x):
        return np.sin(x[0]) - 1.

    def fmax(self):
        return self.fmax

class Sine2:
    def __init__(self):
        self.xmax = 3.614
        self.fmax = self.f([self.xmax])

    def f(self, x):
        return -np.cos(x[0])-np.sin(3*x[0])

    def fmax(self):
        return self.fmax

class DoubleSine:
    def __init__(self, rho1, rho2, tmax):
        self.ep1 = -math.log(rho1, 2)
        self.ep2 = -math.log(rho2, 2)
        self.tmax = tmax
        self.fmax = 0.

    def f(self, x):
        u = 2*math.fabs(x[0]-self.tmax)

        if u == 0:
            return u
        else:
            ew = math.pow(u, self.ep2) - math.pow(u, self.ep1)
            mysin = (math.sin(math.pi*math.log(u, 2))+1)/2.
            return mysin*ew - math.pow(u, self.ep2)

    def fmax(self):
        return self.fmax

class DiffFunc:
    def __init__(self, tmax):
        self.tmax = tmax
        self.fmax = 0.

    def f(self, x):
        u = math.fabs(x[0]-self.tmax)
        if u == 0:
            v = 0
        else:
            v = math.log(u, 2)
        frac = v - math.floor(v)

        if u == 0:
            return u
        elif frac <= 0.5:
            return -math.pow(u, 2)
        else:
            return -math.sqrt(u)

    def fmax(self):
        return self.fmax

class Garland:
    def __init__(self):
        self.fmax = 0.997772313413222

    def f(self, x):
        return 4*x[0]*(1-x[0])*(0.75+0.25*(1-np.sqrt(np.abs(np.sin(60*x[0])))))

    def fmax(self):
        return self.fmax

class Himmelblau:
    def __init__(self):
        self.fmax = 0.

    def f(self, x):
        return -(x[0]**2+x[1]-11.)**2-(x[0]+x[1]**2-7.)**2

    def fmax(self):
        return self.fmax

class Rosenbrock:
    def __init__(self, a, b):
        self.fmax = 0.
        self.a = a
        self.b = b

    def f(self, x):
        return -(self.a-x[0])**2-self.b*(x[1]-x[0]**2)**2

    def fmax(self):
        return self.fmax

class Gramacy1:
    def __init__(self):
        self.xmax = 0.54856343
        self.fmax = self.f([self.xmax])

    def f(self, x):
        return -np.sin(10*np.pi*x[0])/(2*x[0])-(x[0]-1)**4

    def fmax(self):
        return self.fmax

class LossSVM:
    def __init__(self, model, X, y, method, problem):
        self.loss = utils.loss(model, X, y, method, problem)

    def f(self, x):
        return self.loss.evaluateLoss(C=x[0], gamma=x[1])

class LossGBM:
    def __init__(self, model, X, y, method, problem):
        self.loss = utils.loss(model, X, y, method, problem)

    def f(self, x):
        return self.loss.evaluateLoss(learning_rate=x[0], n_estimators=x[1], max_depth=x[2], min_samples_split=x[3])

class LossKNN:
    def __init__(self, model, X, y, method, problem):
        self.loss = utils.loss(model, X, y, method, problem)

    def f(self, x):
        return self.loss.evaluateLoss(n_neighbors=x[0])

class LossMLP:
    def __init__(self, model, X, y, method, problem):
        self.loss = utils.loss(model, X, y, method, problem)

    def f(self, x):
        return self.loss.evaluateLoss(hidden_layer_size=x[0], alpha=x[1])

"""
Function domain partitioning
"""
def std_center(support, support_type):
    """
    Pick the center of a subregion.
    """
    centers = []
    for i in range(len(support)):
        if support_type[i] == 'int':
            a, b = support[i]
            center = (a+b)/2
            centers.append(center)
        elif support_type[i] == 'cont':
            a, b = support[i]
            center = (a+b)/2.
        else:
            raise ValueError('Unsupported variable type.')

    return centers

def std_rand(support, support_type):
    """
    Randomly pick a point in a subregion.
    """
    rands = []
    for i in range(len(support)):
        if support_type[i] == 'int':
            a, b = support[i]
            rand = np.random.randint(a, b+1)
            rands.append(rand)
        elif support_type[i] == 'cont':
            a, b = support[i]
            rand = a + (b-a)*random.random()
            rands.append(rand)
        else:
            raise ValueError('Unsupported variable type.')

    return rands

def std_split(support, support_type, nsplits):
    """
    Split a box uniformly.
    """
    lens = np.array([support[i][1]-support[i][0] for i in range(len(support))])
    max_index = np.argmax(lens)
    max_length = np.max(lens)
    a, b = support[max_index]
    step = max_length/float(nsplits)
    if support_type[max_index] == 'int':
        split = [(a+int(step*i), a+int(step*(i+1))) for i in range(nsplits)]
    elif support_type[max_index] == 'cont':
        split = [(a+step*i, a+step*(i+1)) for i in range(nsplits)]
    else:
        raise ValueError("Unsupported variable type.")

    supports = [None for i in range(nsplits)]
    supports_type = [None for i in range(nsplits)]
    for i in range(nsplits):
        supports[i] = [support[j] for j in range(len(support))]
        supports[i][max_index] = split[i]
        supports_type[i] = support_type

    return supports, supports_type

class Box:
    def __init__(self, f, fmax, nsplits, support, support_type):
        self.f_noised = None
        self.f_mean = f
        self.fmax = fmax
        self.split = None
        self.center = None
        self.rand = None
        self.nsplits = nsplits
        self.support = support
        self.support_type = support_type

    def std_partition(self):
        """
        Standard partitioning of a black box.
        """
        self.center = std_center
        self.rand = std_rand
        self.split = std_split

    def std_noise(self, sigma):
        """
        Stochastic target with Gaussian or uniform noise.
        """
        self.f_noised = lambda x: self.f_mean(x) + sigma*np.random.normal(0, sigma)
        #self.f_noised = lambda x: self.f_mean(x) + sigma*random.random()
    """
    def plot1D(self):
        a, b = self.side
        fig, ax = plt.subplots()
        ax.set_xlim(a, b)
        x = np.array([a+i/10000. for i in range(int((b-a)*10000))])
        y = np.array([self.f_mean([x[i]]) for i in range(int((b-a)*10000))])
        plt.plot(x, y)
        plt.show()

    def plot2D(self):
        # 2D spaced down level curve plot
        x = np.array([(i-600)/100. for i in range(1199)])
        y = np.array([(j-600)/100. for j in range(1199)])
        X, Y = pl.meshgrid(x, y)
        Z = np.array([[self.f_mean([(i-600)/100., (j-600)/100.]) for i in range(1199)] for j in range(1199)])

        im = pl.imshow(Z, cmap=pl.cm.RdBu)
        cset = pl.contour(Z, np.arange(-1, 1.5, 0.2), linewidths=2, cmap=pl.cm.Set2)
        pl.colorbar(im)
        pl.show()

        # 3D plot
        fig = mpl.pyplot.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=pl.cm.RdBu, linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(mpl.ticker.LinearLocator(10))
        ax.zaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        mpl.pyplot.show()
    """