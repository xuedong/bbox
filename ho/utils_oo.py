#!/usr/bin/env python

# This code is based on the original code of Jean-Bastien Grill
# https://team.inria.fr/sequel/Software/POO/

import pylab as pl
import numpy as np
import sys
import math
import random
import pickle

import target
import ho.hoo as hoo
import ho.poo as poo

def std_box(f, fmax, nsplits, sigma, support, support_type):
    box = target.Box(f, fmax, nsplits, support, support_type)
    box.std_partition()
    box.std_noise(sigma)

    return box

# Regret for HOO
def regret_hoo(bbox, rho, nu, alpha, sigma, horizon, update):
    #y = np.zeros(horizon)
    y_cum = [0. for i in range(horizon)]
    y_sim = [0. for i in range(horizon)]
    x_sel = [None for i in range(horizon)]
    htree = hoo.HTree(bbox.support, bbox.support_type, None, 0, rho, nu, bbox)
    cum = 0.

    for i in range(horizon):
        if update and alpha < math.log(i+1) * (sigma ** 2):
            alpha += 1
            htree.update(alpha)
        x, _, _ = htree.sample(alpha)
        cum += bbox.fmax - bbox.f_mean(x)
        y_cum[i] = cum/(i+1)
        x_sel[i] = x
        z = random.choice(x_sel[0:(i+1)])
        y_sim[i] = bbox.fmax - bbox.f_mean(z)

    return y_cum, y_sim, x_sel

def loss_hoo(bbox, rho, nu, alpha, sigma, horizon, update):
    losses = [0. for i in range(horizon)]
    htree = hoo.HTree(bbox.support, bbox.support_type, None, 0, rho, nu, bbox)
    best = -float("inf")

    for i in range(horizon):
        if update and alpha < math.log(i+1) * (sigma ** 2):
            alpha += 1
            htree.update(alpha)
        x, _, _ = htree.sample(alpha)
        current = bbox.f_mean(x)
        if current > best:
            best = current
            losses[i] = best
        else:
            losses[i] = best

    return losses
"""
def regret_hct(bbox, rho, nu, c, c1, delta, sigma, horizon):
    y_cum = [0. for i in range(horizon)]
    y_sim = [0. for i in range(horizon)]
    x_sel = [None for i in range(horizon)]
    hctree = hct.HCTree(bbox.support, None, 0, rho, nu, 1, 1, bbox)
    cum = 0.

    for i in range(1, horizon+1):
        tplus = int(2**(math.ceil(math.log(i))))
        dvalue = min(c1*delta/tplus, 0.5)

        if i == tplus:
            hctree.update(c, dvalue)
        x, _, _ = hctree.sample(c, dvalue)
        cum += bbox.fmax - bbox.f_mean(x)
        y_cum[i-1] = cum/i
        x_sel[i-1] = x
        z = random.choice(x_sel[0:i])
        y_sim[i-1] = bbox.fmax - bbox.f_mean(z)

    return y_cum, y_sim, x_sel
"""
def regret_poo(bbox, rhos, nu, alpha, horizon, epoch):
    y_cum = [[0. for j in range(horizon)] for i in range(epoch)]
    y_sim = [[0. for j in range(horizon)] for i in range(epoch)]
    x_sel = [[None for j in range(horizon)] for i in range(epoch)]
    for i in range(epoch):
        ptree = poo.PTree(bbox.support, bbox.support_type, None, 0, rhos, nu, bbox)
        count = 0
        length = len(rhos)
        cum = [0.] * length
        emp = [0.] * length
        smp = [0] * length

        while count < horizon:
            for k in range(length):
                x, noisy, existed = ptree.sample(alpha, k)
                cum[k] += bbox.fmax - bbox.f_mean(x)
                count += existed
                emp[k] += noisy
                smp[k] += 1

                if existed and count <= horizon:
                    best_k = max(range(length), key=lambda x: (-float("inf") if smp[k] == 0 else emp[x]/smp[k]))
                    y_cum[i][count-1] = 1 if smp[best_k] == 0 else cum[best_k]/float(smp[best_k])
                    x_sel[i][count-1] = x
                    z = random.choice(x_sel[i][0:count])
                    y_sim[i][count-1] = bbox.fmax - bbox.f_mean(z)

    return y_cum, y_sim, x_sel

def loss_poo(bbox, rhos, nu, alpha, horizon, epoch):
    losses = [[0. for j in range(horizon)] for i in range(epoch)]
    for i in range(epoch):
        ptree = poo.PTree(bbox.support, bbox.support_type, None, 0, rhos, nu, bbox)
        count = 0
        length = len(rhos)
        cum = [0.] * length
        emp = [0.] * length
        smp = [0] * length

        while count < horizon:
            for k in range(length):
                x, noisy, existed = ptree.sample(alpha, k)
                cum[k] += bbox.f_mean(x)
                count += existed
                emp[k] += noisy
                smp[k] += 1

                if existed and count <= horizon:
                    best_k = max(range(length), key=lambda x: (-float("inf") if smp[k] == 0 else emp[x]/smp[k]))
                    losses[i][count-1] = -1 if smp[best_k] == 0 else cum[best_k]/float(smp[best_k])

    return losses

# Plot regret curve
def show(path, epoch, horizon, rhos_hoo, rhos_poo, delta):
    data = [None for k in range(epoch)]
    for k in range(epoch):
        with open(path+"HOO"+str(k+1), 'rb') as file:
            data[k] = pickle.load(file)
    with open(path+"POO", 'rb') as file:
        data_poo = pickle.load(file)
    #with open(path+"GPUCB_DIRECT", 'rb') as file:
    #    data_ucb_direct = pickle.load(file)
    #with open(path+"GPUCB_LBFGS", 'rb') as file:
    #    data_ucb_lbfgs = pickle.load(file)
    #with open(path+"EI", 'rb') as file:
    #    data_ei = pickle.load(file)
    #with open(path+"PI", 'rb') as file:
    #    data_pi = pickle.load(file)
    #with open(path+"THOMP", 'rb') as file:
    #    data_thomp = pickle.load(file)

    length_hoo = len(rhos_hoo)
    length_poo = len(rhos_poo)
    rhostoshow = [int(length_hoo*k/4.) for k in range(4)]
    #rhostoshow = [0, 6, 12, 18]
    #rhostoshow = [0, 1, 3, 7, 15]
    style = [[5,5], [1,3], [5,3,1,3], [5,2,5,2,5,10]]
    #style = [[5,5], [1,3], [5,3,1,3], [5,2,5,2,5,10], [3, 1]]

    means = [[sum([data[k][j][i] for k in range(epoch)])/float(epoch) for i in range(horizon)] for j in range(length_hoo)]
    devs = [[math.sqrt(sum([(data[k][j][i]-means[j][i])**2 for k in range(epoch)])/(float(epoch)*float(epoch-1))) for i in range(horizon)] for j in range(length_hoo)]

    means_poo = [sum([data_poo[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]

    #means_ucb_direct = [sum([data_ucb_direct[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]

    #means_ucb_lbfgs = [sum([data_ucb_lbfgs[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]

    #means_ei = [sum([data_ei[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]

    #means_pi = [sum([data_pi[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]

    #means_thomp = [sum([data_thomp[u][v]/float(epoch) for u in range(epoch)]) for v in range(horizon)]

    X = np.array(range(horizon))
    for i in range(len(rhostoshow)):
        k = rhostoshow[i]
        label__ = r"$\mathtt{HOO}, \rho = " + str(float(k)/float(length_hoo)) + "$"
        pl.plot(X, np.array(means[k]), label=label__, dashes=style[i])
    pl.plot(X, np.array(means_poo), label=r"$\mathtt{POO}$")
    #pl.plot(X, np.array(means_ucb_direct), label=r"$\mathtt{GPUCB-DIRECT}$", color='blue')
    #pl.plot(X, np.array(means_ucb_lbfgs), label=r"$\mathtt{GPUCB-LBFGS}$", color='green')
    #pl.plot(X, np.array(means_ei), label=r"$\mathtt{EI}$", color='cyan')
    #pl.plot(X, np.array(means_pi), label=r"$\mathtt{PI}$", color='magenta')
    #pl.plot(X, np.array(means_thomp), label=r"$\mathtt{THOMPSON}$", color='magenta')
    pl.legend()
    pl.xlabel("number of evaluations")
    pl.ylabel("simple regret")
    pl.show()

    X = np.array(map(math.log, range(horizon)[1:]))
    for i in range(len(rhostoshow)):
        k = rhostoshow[i]
        label__ = r"$\mathtt{HOO}, \rho = " + str(float(k)/float(length_hoo)) + "$"
        pl.plot(X, np.array(map(math.log, means[k][1:])), label=label__, dashes=style[i])
    pl.plot(X, np.array(map(math.log, means_poo[1:])), label=r"$\mathtt{POO}$")
    #pl.plot(X, np.array(map(math.log, means_ucb_direct[1:])), label=r"$\mathtt{GPUCB-DIRECT}$", color='blue')
    #pl.plot(X, np.array(map(math.log, means_ucb_lbfgs[1:])), label=r"$\mathtt{GPUCB-LBFGS}$", color='green')
    #pl.plot(X, np.array(map(math.log, means_ei[1:])), label=r"$\mathtt{EI}$", color='cyan')
    #pl.plot(X, np.array(map(math.log, means_pi[1:])), label=r"$\mathtt{PI}$", color='magenta')
    #pl.plot(X, np.array(map(math.log, means_thomp[1:])), label=r"$\mathtt{THOMPSON}$", color='magenta')
    pl.legend(loc=3)
    pl.xlabel("number of evaluations (log-scale)")
    pl.ylabel("simple regret")
    pl.show()

    X = np.array([float(j)/float(length_hoo-1) for j in range(length_hoo)])
    Y = np.array([means[j][horizon-1] for j in range(length_hoo)])
    Z1 = np.array([means[j][horizon-1]+math.sqrt(2*(devs[j][horizon-1] ** 2) * math.log(1/delta)) for j in range(length_hoo)])
    #Z1 = np.array([means[j][horizon-1]+2*devs[j][horizon-1] for j in range(length_hoo)])
    Z2 = np.array([means[j][horizon-1]-math.sqrt(2*(devs[j][horizon-1] ** 2) * math.log(1/delta)) for j in range(length_hoo)])
    #Z2 = np.array([means[j][horizon-1]-2*devs[j][horizon-1] for j in range(length_hoo)])
    pl.plot(X, Y)
    pl.plot(X, Z1, color="green")
    pl.plot(X, Z2, color="green")
    pl.xlabel(r"$\rho$")
    pl.ylabel("simple regret after " + str(horizon) + " evaluations")
    pl.show()

    X = np.array([float(j)/float(length_hoo-1) for j in range(length_hoo)])
    Y = np.array([means[j][horizon-1] for j in range(length_hoo)])
    E = np.array([math.sqrt(2*(devs[j][horizon-1] ** 2) * math.log(1/delta)) for j in range(length_hoo)])
    #E = np.array([2*devs[j][horizon-1] for j in range(length_hoo)])
    pl.errorbar(X, Y, yerr=E, color='black', errorevery=3)
    pl.xlabel(r"$\rho$")
    pl.ylabel("simple regret after " + str(horizon) + " evaluations")
    pl.show()

# How to choose rhos to be used
def get_rhos(nsplits, rhomax, horizon):
    dmax = math.log(nsplits)/math.log(1./rhomax)
    n = 0
    N = 1
    rhos = np.array([rhomax])

    while n < horizon:
        if n == 0 or n == 1:
            threshold = -float("inf")
        else:
            threshold = 0.5 * dmax * math.log(n/math.log(n))

        while N <= threshold:
            for i in range(N):
                rho_current = math.pow(rhomax, 2.*N/(2*(i+1)))
                rhos = np.append(rhos, rho_current)
            n = 2*n
            N = 2*N

        n = n+N

    return np.unique(np.sort(rhos))
