#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

import numpy as np
import math
import random
#import operator as op
import pylab as pl
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

"""
Target functions
"""
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

"""
Function domain partitioning
"""
def std_center(support):
    return [(support[i][0]+support[i][1])/2. for i in range(len(support))]

def std_rand(support):
    return [(support[i][0]+(support[i][1]-support[i][0])*random.random()) for i in range(len(support))]

def std_split(support, nsplits):
    lens = np.array([support[i][1]-support[i][0] for i in range(len(support))])
    max_index = np.argmax(lens)
    max_length = np.max(lens)
    a, b = support[max_index]
    step = max_length/float(nsplits)
    split = [(a+step*i, a+step*(i+1)) for i in range(nsplits)]

    supports = [None for i in range(nsplits)]
    for i in range(nsplits):
        supports[i] = [support[j] for j in range(len(support))]
        supports[i][max_index] = split[i]

    return supports

class Box:
    def __init__(self, f, fmax, nsplits, dim, side):
        self.f_noised = None
        self.f_mean = f
        self.fmax = fmax
        self.split = None
        self.center = None
        self.rand = None
        self.nsplits = nsplits
        self.dim = dim
        self.side = side

    def std_partition(self):
        self.center = std_center
        self.rand = std_rand
        self.split = std_split
        self.support = []
        for i in range(self.dim):
            self.support.append(self.side)

    def std_noise(self, sigma):
        self.f_noised = lambda x: self.f_mean(x) + sigma*np.random.normal(0, sigma)
        #self.f_noised = lambda x: self.f_mean(x) + sigma*random.random()

    def plot1D(self):
        """
        Plot for 1D function.
        """
        x = np.array([(i+1)/10000. for i in range(9999)])
        y = np.array([self.f_mean([(i+1)/10000.]) for i in range(9999)])
        pl.plot(x, y)
        pl.show()

    def plot2D(self):
        """
        Plot for 2D function
        """
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
