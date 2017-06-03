#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

import random
import numpy as np
import math

import ho.hoo as hoo

class PTree:
    def __init__(self, support, support_type, father, depth, rhos, nu, box):
        self.bvalues = np.array([float("inf")] * len(rhos))
        self.uvalues = np.array([float("inf")] * len(rhos))
        self.tvalues = np.array([0] * len(rhos))
        self.rewards = np.array([0.] * len(rhos))
        self.noisy = None
        self.evaluated = None
        self.support = support
        self.support_type = support_type
        self.father = father
        self.depth = depth
        self.rhos = rhos
        self.nu = nu
        self.box = box
        self.children = []
        self.hoos = [hoo.HTree(support, support_type, father, depth, rhos[k], nu, box) for k in range(len(rhos))]

    def add_children(self):
        supports, supports_type = self.box.split(self.support, self.support_type, self.box.nsplits)

        self.children = [PTree(supports[i], supports_type[i], self, self.depth + 1, self.rhos, self.nu, self.box) for i in range(len(supports))]

    def explore(self, k):
        if self.tvalues[k] == 0:
            return self
        elif not(self.children):
            self.add_children()
            return random.choice(self.children)
        else:
            return max(self.children, key=lambda x: x.bvalues[k]).explore(k)

    def explore_bis(self, k):
        return self.hoos[k].explore()

    def update_node(self, alpha, k):
        if self.tvalues[k] == 0:
            self.uvalues[k] = float("inf")
        else:
            mean = float(self.rewards[k])/float(self.tvalues[k])
            hoeffding = math.sqrt(2.*alpha/float(self.tvalues[k]))
            variation = self.nu * math.pow(self.rhos[k], self.depth)

            self.uvalues[k] = mean + hoeffding + variation

    def update_node_bis(self, alpha, k):
        self.hoos[k].update_node(alpha)

    def update_path(self, reward, alpha, k):
        self.rewards[k] += reward
        self.tvalues[k] += 1
        self.update_node(alpha, k)

        if not(self.children):
            self.bvalues[k] = self.uvalues[k]
        else:
            self.bvalues[k] = min(self.uvalues[k], max([child.bvalues[k] for child in self.children]))

        if self.father is not(None):
            self.father.update_path(reward, alpha, k)

    def update_path_bis(self, reward, alpha, k):
        self.hoos[k].update_path(reward, alpha)

    def update_all(self, alpha):
        for k in range(len(rhos)):
            self.update_node(alpha, k)

        if not(self.children):
            self.bvalues = self.uvalues
        else:
            for child in self.children:
                child.update_all(alpha)
            for k in range(len(rhos)):
                self.bvalues[k] = min(self.uvalues[k], max([child.bvalues[k] for child in self.children]))

    def update_all_bis(self, alpha):
        for k in range(len(rhos)):
            self.hoos[k].update(alpha)

    def sample(self, alpha, k):
        leaf = self.explore(k)
        existed = False

        if leaf.noisy is None:
            x = self.box.rand(leaf.support, leaf.support_type)
            leaf.evaluated = x
            leaf.noisy = self.box.f_noised(x)
            existed = True
        leaf.update_path(leaf.noisy, alpha, k)

        return leaf.evaluated, leaf.noisy, existed

    def sample_bis(self, alpha, k):
        return self.hoos[k].sample(alpha)
