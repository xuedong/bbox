#!/usr/bin/env python

import random
import numpy as np
import math

class HOO:
    def __init__(self):
        self.nodes = np.array([(0, 1)])
        self.path = np.array([(0, 1)])
        self.bvalues = np.array([0])
        self.uvalues = np.array([0])
        self.tvalues = np.array([0])

    def get_leaf(self):
        leaf = random.choice(self.nodes[1:])
        return leaf

    def get_child(self):

