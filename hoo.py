#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

import random
import numpy as np
import math

class HTree:
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
        supports = self.box.split(self.support)

        self.children = [HTree(s, self, self.depth + 1, self.rho, self.nu, self.box) for s in supports]

    def explore(self):
        if self.tvalue == 0:
            return self
        elif not(self.children):
            self.add_children()
            return random.choice(self.children)
        else:
            return max(self.children, key=lambda x: x.bvalue).explore()

    def update_node(self, alpha):
        if self.tvalue == 0:
            self.uvalue = float('inf')
        else:
            mean = float(self.reward)/float(self.tvalue)
            hoeffding = math.sqrt(2.*alpha/float(self.tvalue))
            variation = self.nu * math.pow(self.rho, self.depth)

            self.uvalue = mean + hoeffding + variation

    def update_path(self, reward, alpha):
        self.reward += reward
        self.tvalue += 1
        self.update_node(alpha)

        if not(self.children):
            self.bvalue = self.uvalue
        else:
            self.bvalue = min(self.uvalue, max([child.bvalue for child in self.children]))

        if self.father is not(None):
            self.father.update_path(reward, alpha)

    def update(self, alpha):
        self.update_node(alpha)
        if not(self.children):
            self.bvalue = self.uvalue
        else:
            for child in self.children:
                child.update(alpha)
                self.bvalue = min(self.uvalue, max([child.bvalue for child in self.children]))

    def sample(self, alpha):
        leaf = self.explore()

        if leaf.noisy is None:
            x = self.box.rand(leaf.support)
            leaf.evaluated = x
            leaf.noisy = self.box.f_noised(x)
        leaf.update_path(leaf.noisy, alpha)

        return leaf.evaluated
