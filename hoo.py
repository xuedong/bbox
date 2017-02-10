#!/usr/bin/env python

import random
import numpy as np
import math

class HTree:
    def __init__(self, support, father, depth, rho, box):
        self.bvalue = float('inf')
        self.uvalue = float('inf')
        self.support = support
        self.father = father
        self.depth = depth
        self.rho = rho
        self.box = box
        self.children = []

    def get_leaf(self):
        leaf = random.choice(self.nodes[1:])
        return leaf

    def add_children(self):
        supports = self.box.split(self.support)
        self.children = [Tree(s, self, self.depth + 1, self.rho, self.box) for s in supports]
