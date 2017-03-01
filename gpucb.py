#!/usr/bin/env python

# This code is based on the original code of Emile Contal in Matlab
# econtal.person.math.cnrs.fr/software

import numpy as np
import math

def gpucb(var_post, log_prob, nb_tests):
    return math.sqrt(2.*(log_prob+math.log(nb_tests))*var_post)
