#!/usr/bin/env python

# This code is based on the original code of Emile Contal in Matlab
# econtal.person.math.cnrs.fr/software

import numpy as np
import matplotlib.pyplot as plt
import math
import time

import basis
import gp
import acquisition

from reggie import make_gp, MCMC
from ezplot import figure, show

from pybo import inits
from pybo import policies
from pybo import solvers
from pybo import recommenders


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

def sample(d=1, side=(0, 40), ns=1000, nt=10, kernel=basis.kernel_se_norm, basis=basis.basis_none, noise=0.01, posterior=True, verbose=True, plot=True, bbox=None):
    a, b = side
    size = b-a
    Xs = a + size*np.random.rand(d, ns)
    #print(Xs)
    Xt = Xs[0:d, 0:nt]
    print(Xt[0][4])
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
    if bbox:
        f = bbox.f_noised
        Yt = np.array([[f([Xt[0][i]])] for i in np.arange(nt)]).T
    else:
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
            #print(BI.var.shape)
            #print(Z[:, np.newaxis].T)
            ax.fill_between(np.squeeze(np.asarray(Z[:, np.newaxis])),
                            np.squeeze(np.asarray(BI.mu[IZ]+2*BI.var[IZ][:, np.newaxis])),
                            np.squeeze(np.asarray(BI.mu[IZ]-2*BI.var[IZ][:, np.newaxis])),
                            facecolor='cyan')

            plt.show()

    return f, Xs, Ys, Xt, Yt, Kss

def bo(f, Xt, Yt, Xs, iterations, algo='gpucb', noise=0.01, u=3, kernel=basis.kernel_se_norm, basis=basis.basis_none, plot=True, verbose=True, bbox=True):
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

    if plot:
        fig = figure(figsize=(10, 6))

    for i in range(iterations):
        BI.posterior(Kts, DKss, Ht, Hs)
        u += math.log((i+1)**2*math.pi**2/6)

        if algo is 'gpucb':
            ucb = acquisition.gpucb(BI.var, u, ns)
            target = BI.mu + ucb.T
            #print(BI.mu.shape)
            #print(ucb.shape)
            #print(target.shape)
            ymax = np.max(target)
            #print(ymax)
            xmax = np.argmax(target)

        np.append(queries, xmax)
        xt = np.array([xmax])
        if bbox:
            yt = f([Xs[0][xmax]])
        else:
            yt = f(xt)
        Xt = np.concatenate((Xt, Xs[:, xt[-1]][:, np.newaxis]), 1)
        Yt = np.append(Yt, yt)

        # GP update
        Ht = np.concatenate((Ht, basis(Xt[-1, :])), 0)
        Ktt12 = kernel(Xt[:, 0:-1], Xt[:, -1][:, np.newaxis])
        Ktt22 = kernel(Xt[:, -1][:, np.newaxis], Xt[:, -1][:, np.newaxis])
        BI.inference_update(Ktt12, Ktt22, Yt)
        Kts = np.concatenate((Kts, kernel(Xt[:, -1][:, np.newaxis], Xs)), 0)

        if plot:
            fig.clear()
            ax1 = fig.add_subplotspec((2, 2), (0, 0), hidex=True)
            ax2 = fig.add_subplotspec((2, 2), (1, 0), hidey=True, sharex=ax1)

            Z = np.sort(Xs[0, :])
            IZ = np.argsort(Xs[0, :])
            ax1.plot(Z, BI.mu[IZ], '-r')

            target_plot = np.min(BI.mu) + (target-np.min(target))*(np.max(BI.mu)-np.min(BI.mu))/(np.max(target)-np.min(target))
            #ax1.plot(Z, target_plot[IZ], 'k')
            ax1.plot(Xt[0, :], Yt, 'x')
            ax1.fill_between(np.squeeze(np.asarray(Z[:, np.newaxis])),
                            np.squeeze(np.asarray(BI.mu[IZ]+2*BI.var.T[IZ])),
                            np.squeeze(np.asarray(BI.mu[IZ]-2*BI.var.T[IZ])),
                            facecolor='cyan')

            fig.canvas.draw()
            time.sleep(1)
            show(block=False)

        if verbose:
            print(str(i+1) + '. max: ' + str(np.max(Yt)) + ', target: ' + str(ymax) + ', observed: ' + str(yt))

    if plot:
        plt.show()

    return queries, Yt

def animated(f, bounds, plot, verbose, horizon):
    xopt = f.xmax
    fopt = f.fmax
    x = np.linspace(bounds[0][0], bounds[0][1], 500)

    X = list(inits.init_latin(bounds, 3))
    Y = [f.f([x_])[0] for x_ in X]
    F = []

    model = make_gp(0.01, 1.9, 0.1, 0)
    model.add_data(X, Y)

    model.params['like.sn2'].set_prior('uniform', 0.005, 0.015)
    model.params['kern.rho'].set_prior('lognormal', 0, 100)
    model.params['kern.ell'].set_prior('lognormal', 0, 10)
    model.params['mean.bias'].set_prior('normal', 0, 20)

    model = MCMC(model, n=20, rng=None)

    if plot:
        fig = figure(figsize=(10, 6))

    if verbose:
        print('Bayesian update...')

    for i in range(horizon):
        if verbose:
            print('Step '+str(i+1))

        # get acquisition function (or index)
        index = policies.EI(model, bounds, X, xi=0.1)

        # get the recommendation and the next query
        xbest = recommenders.best_incumbent(model, bounds, X)
        xnext, _ = solvers.solve_lbfgs(index, bounds)
        ynext = f.f([xnext])[0]

        # evaluate the posterior before updating the model for plotting
        mu, s2 = model.predict(x[:, None])

        # record our data and update the model
        X.append(xnext)
        Y.append(ynext)
        F.append(f.f([xbest])[0])
        model.add_data(xnext, ynext)

        if plot:
            # plot everything
            fig.clear()
            ax1 = fig.add_subplotspec((2, 2), (0, 0), hidex=True)
            ax2 = fig.add_subplotspec((2, 2), (1, 0), hidey=True, sharex=ax1)
            ax3 = fig.add_subplotspec((2, 2), (0, 1), rowspan=2)

            # plot the posterior and data
            ax1.plot_banded(x, mu, 2*np.sqrt(s2))
            ax1.scatter(np.ravel(X), Y)
            ax1.axvline(xbest)
            ax1.axvline(xnext, color='g')
            ax1.set_ylim(-6, 3)
            ax1.set_title('current model (xbest and xnext)')

            # plot the acquisition function
            ax2.plot_banded(x, index(x[:, None]))
            ax2.axvline(xnext, color='g')
            ax2.set_xlim(*bounds)
            ax2.set_title('current policy (xnext)')

            # plot the latent function at recomended points
            ax3.plot(F)
            ax3.axhline(fopt)
            ax3.set_ylim(0.4, 0.9)
            ax3.set_title('value of recommendation')

            # draw
            fig.canvas.draw()
            show(block=False)

        i += 1
