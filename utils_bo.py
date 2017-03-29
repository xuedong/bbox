#!/usr/bin/env python

# This code is based on the original code of Emile Contal in Matlab
# econtal.person.math.cnrs.fr/software

import numpy as np
import matplotlib.pyplot as plt
import math

import basis
import gp
import acquisition

def cummax(R):
    d = R.shape[0]
    n = R.shape[1]
    M = np.zeros((d, n))

    maxi = R[:, 0]
    M[:, 0] = maxi

    for i in range(1, n):
        maxi = np.maximum(maxi, R[:, i])
        #print(maxi)
        M[:, i] = maxi
        #print(M[:, i])

    return M

def solve_chol(R, Y):
    return np.linalg.solve(R, np.linalg.solve(R.T, Y))

def approx_chol(X):
    m = np.min(X)
    m = max(np.finfo(float).eps, m*1e-14)
    gap = m
    I = np.eye(X.shape[0], X.shape[1])
    ok = False

    while not(ok):
        try:
            R = np.linalg.cholesky(X)
            ok = True
        except np.linalg.linalg.LinAlgError as e:
            X = X + gap*I
            if gap > 1e6*m:
                print('Warning, adding '+str(gap)+' for cholpsd')
            gap *= 10

    return R.conj().T

def sample(d=1, size=40, ns=1000, nt=10, kernel=basis.kernel_se_norm, basis=basis.basis_none, noise=0.01, posterior=True, verbose=True, plot=True):
    Xs = size*np.random.rand(d, ns)
    #print(Xs)
    Xt = Xs[0:d, 0:nt]
    #print(Xt)
    Kss = kernel(Xs, Xs)
    #print(K)
    B = basis(Xs)
    Ys = np.dot(approx_chol(Kss).T, np.random.randn(ns, 1))
    #print(Ys)
    if B:
        Ys = Ys + np.dot(B, np.random.randn(B.shape[1], 1))
    Ys = Ys.T
    #S = np.squeeze(np.asarray(Ys))
    S = np.reshape(Ys, (Ys.shape[0]*Ys.shape[1], 1))
    #print(S.shape)
    f = lambda X: S[X] + noise*np.random.randn(X.shape[0], 1)
    Yt = f(np.arange(nt)).T
    #print(Yt)

    if posterior:
        if plot:
            fig, (ax) = plt.subplots(1, 1)
            ax.plot(Xt, Yt, 'd', color="blue")
            #ax.plot(Xs, Ys, 'bo', color="red")

        if verbose:
            print("Bayesian inference...")

        Ktt = Kss[0:nt, 0:nt]
        Ht = basis(Xs)
        BI = gp.GP(noise)
        BI.inference(Ktt, Yt.T, Ht)

        if verbose:
            print("Prediction...")

        Kts = Kss[0:nt,0:ns]
        Hs = basis(Xs)
        DKss = np.diag(Kss)
        BI.posterior(Kts, DKss, Ht, Hs)

        if plot:
            Z = np.sort(Xs[0, :])
            IZ = np.argsort(Xs[0, :])
            ax.plot(Z, BI.mu[IZ], '-r')
            #print(BI.mu[IZ].T)
            #print(BI.var[IZ].shape)
            #print(Z[:, np.newaxis].T)
            ax.fill_between(np.squeeze(np.asarray(Z[:, np.newaxis])),
                            np.squeeze(np.asarray(BI.mu[IZ]+2*BI.var[IZ][:, np.newaxis])),
                            np.squeeze(np.asarray(BI.mu[IZ]-2*BI.var[IZ][:, np.newaxis])),
                            facecolor='cyan')

            plt.show()

    return f, Xs, Ys, Xt, Yt, Kss

def bo(f, Xt, Yt, Xs, iterations, algo='gpucb', noise=0.01, u=3, kernel=basis.kernel_se_norm, basis=basis.basis_none, plot=True, verbose=True):
    target_plot = np.array([])
    d, ns = Xs.shape
    queries = np.array([])
    Ht = basis(Xt)
    Hs = basis(Xs)
    Ktt = kernel(Xt, Xt)
    Kts = kernel(Xt, Xs)
    DKss = kernel(Xs, 'diag')

    BI = gp.GP(noise)
    BI.inference(Ktt, Yt.T, Ht)

    for i in range(iterations):
        BI.posterior(Kts, DKss, Ht, Hs)
        u += math.log((i+1)**2*math.pi**2/6)

        if algo is 'gpucb':
            ucb = acquisition.gpucb(BI.var, u, ns)
            target = BI.mu + ucb.T
            ymax = np.max(target)
            xmax = np.argmax(target)

        np.append(queries, xmax)
        xt = np.array([xmax])
        yt = f(xt)
        Xt = np.concatenate((Xt, Xs[:, xt[-1]][:, np.newaxis]), 1)
        Yt = np.append(Yt, yt)

        # GP update
        Ht = np.concatenate((Ht, basis(Xt[-1, :])), 0) 

    return queries, Yt
