#!/usr/bin/env python

import random
import numpy as np
import math

class HCTree:
    def __init__(self, support, father, depth, rho, nu, tvalue, tau, box):
        self.bvalue = float('inf')
        self.uvalue = float('inf')
        self.tvalue = tvalue
        self.tau = tau
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

    def add_children(self, c, dvalue):
        supports = self.box.split(self.support, self.box.nsplits)
        #print(supports)

        tau = c**2 * math.log(1./dvalue) * self.rho**(-2*(self.depth+1))/(self.nu**2)
        self.children = [HCTree(s, self, self.depth + 1, self.rho, self.nu, 0, tau, self.box) for s in supports]

    def explore(self, c, dvalue):
        if self.tvalue < self.tau:
            return self
        elif not(self.children):
            self.add_children(c, dvalue)
            return random.choice(self.children)
        else:
            return max(self.children, key=lambda x: x.bvalue).explore(c, dvalue)

    def update_node(self, c, dvalue):
        if self.tvalue == 0:
            self.uvalue = float('inf')
        else:
            mean = float(self.reward)/float(self.tvalue)
            hoeffding = c * math.sqrt(math.log(1./dvalue)/float(self.tvalue))
            variation = self.nu * math.pow(self.rho, self.depth)

            self.uvalue = mean + hoeffding + variation

    def update_path(self, reward, c, dvalue):
        if not(self.children):
            self.reward += reward
            self.tvalue += 1
        self.update_node(c, dvalue)

        if not(self.children):
            self.bvalue = self.uvalue
        else:
            self.bvalue = min(self.uvalue, max([child.bvalue for child in self.children]))

        if self.father is not(None):
            self.father.update_path(reward, c, dvalue)

    def update(self, c, dvalue):
        self.update_node(c, dvalue)
        if not(self.children):
            self.bvalue = self.uvalue
        else:
            for child in self.children:
                child.update(c, dvalue)
                self.bvalue = min(self.uvalue, max([child.bvalue for child in self.children]))

    def sample(self, c, dvalue):
        leaf = self.explore(c, dvalue)
        existed = False

        if leaf.noisy is None:
            x = self.box.center(leaf.support)
            leaf.evaluted = x
            leaf.noisy = self.box.f_noised(x)
            existed = True
        leaf.update_path(leaf.noisy, c, dvalue)

        return leaf.evaluted, leaf.noisy, existed
