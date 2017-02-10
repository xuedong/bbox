#!/usr/bin/env python

import pylab as pl
import numpy as np
import os
import sys
import math
import random

import target
import hoo

# Macros
HORIZON = 1000
RHOMAX = 0.9
VAR = 0.1

# Hyperparameters
alpha_ = math.log(HORIZON)
nu_ = 1.

def std_box(f, fmax):
    box = target.Box(f, fmax)
    box.std_noise(VAR)
    box.std_partition()

    return box

# First test
f1 = target.DoubleSine(0.3, 0.8, 0.5)
bbox1 = std_box(f1.f, f1.fmax)
bbox1.plot()
