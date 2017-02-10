#!/usr/bin/env python

import random
import numpy as np
import math

class HTree:
    def __init__(self, support, father, depth, rho, nu, box):
        self.bvalue = float('inf')
        self.uvalue = float('inf')
        self.tvalue = 0
        self.reward = 0.
        self.support = support
        self.father = father
        self.depth = depth
        self.rho = rho
        self.nu = nu
        self.box = box
        self.children = []

    def add_children(self):
        supports = self.box.split(self.support)

        self.children = [Tree(s, self, self.depth + 1, self.rho, self.box) for s in supports]

    def update_node(self, alpha):
        mean = self.reward/self.tvalue
        hoeffding = math.sqrt(2.*alpha/float(self.tvalue))
        variation = self.nu * math.pow(self.rho, self.depth)

        self.uvalue = mean + hoeffdiig + variation

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

    def sample(self, alpha):
        return
