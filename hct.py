#!/usr/bin/env python

import random
import numpy as np
import math

class HCTree:
    def __init__(self, support, father, depth, rho, nu, box):
        self.bvalue = float('inf')
        self.uvalue = float('inf')
        self.tvalue = 0
        self.reward = 0.
        self.noisy = None
        self.evaluated = None
        self.support = support
        self.father = father
        self.depth = depth
        self.rho = rho
        self.nu = nu
        self.box = box
        self.children = []

    def add_children(self):
        supports = self.box.split(self.support, self.box.nsplits)
        #print(supports)

        self.children = [HCTree(s, self, self.depth + 1, self.rho, self.nu, self.box) for s in supports]


