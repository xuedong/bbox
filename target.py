#!/usr/bin/env python

import numpy as np
import math
import random
import pylab as pl

"""
Target function
"""
class DoubleSine:
    def __init__(self, rho1, rho2, tmax):
        self.ep1 = -math.log(rho1, 2)
        self.ep2 = -math.log(rho2, 2)
        self.tmax = tmax
        self.fmax = 0.

    def f(self, x):
        u = 2*math.fabs(x-self.tmax)
        if u == 0:
            return u
        else:
            ew = math.pow(u, self.ep2) - math.pow(u, self.ep1)
            mysin = (math.sin(math.pi*math.log(u, 2))+1)/2.
            return mysin*ew - math.pow(u, self.ep2)

    def fmax(self):
        return self.fmax

"""
Function domain partitioning
"""
class Box:
    def __init__(self, f, fmax):
        self.f_noised = None
        self.f_mean = f
        self.fmax = fmax
        self.split = None
        self.center = None
        self.rand = None

    def std_center(self, support):
        a, b = support

        return (a+b)/2.

    def std_rand(self, support):
        a, b = support

        return a + (b-a)*random.random()

    def std_split(self, support):
        a, b = support
        m = (a+b)/2.

        return [(a, m), (m, b)]

    def std_noise(self, var):
        self.f_noised = lambda x: f_mean(x) + var*random.random()

    def std_partition(self):
        self.center = self.std_center
        self.rand = self.std_rand
        self.split = self.std_split

    def plot(self):
        x = np.array([i/10000. for i in range(10000)])
        y = np.array([self.f_mean(i/10000.) for i in range(10000)])
        pl.plot(x, y)
        pl.show()
