#!/usr/bin/env python

# This code is based on the original code of Emile Contal in Matlab
# econtal.person.math.cnrs.fr/software

import numpy as np
import basis
import choldate

from utils_bo import solve_chol
from utils_bo import approx_chol

class GP:
    def __init__(self, noise):
        self.noise = noise

    def inference(self, K, Y, H=np.array([])):
        C = K + self.noise*np.eye(K.shape[0], K.shape[1])
        self.RC = approx_chol(C)

        if H:
            self.RHCH = approx_chol(np.dot(H.T, solve_chol(self.RC, H)))
            self.beta = solve_chol(self.RHCH, np.dot(H.T, solve_chol(self.RC, Y)))
            self.invCY = solve_chol(self.RC, Y-np.dot(H, self.beta))
        else:
            self.invCY = solve_chol(self.RC, Y)
            self.beta = np.array([])
            self.RHCH = np.array([])

    def inference_update(self, K12, K22, Y, H=np.array([])):
        n1, n2 = K12.shape

        C12 = K12
        C22 = K22 + self.noise*np.eye(K22.shape[0], K22.shape[1])

        RC11 = self.RC
        RC12 = np.linalg.solve(RC11.T, C12)
        RC22 = np.linalg.cholesky(C22 - np.dot(RC12.T, RC12)).conj().T
        RC1 = np.concatenate((RC11, RC12), 1)
        RC2 = np.concatenate((np.zeros(RC12.T.shape), RC22), 1)
        self.RC = np.concatenate((RC1, RC2), 0)

        if H:
            self.RHCH = approx_chol(np.dot(H.T, solve_chol(self.RC, H)))
            self.beta = solve_chol(self.RHCH, np.dot(H.T, solve_chol(self.RC, Y)))
            self.invCY = solve_chol(self.RC, Y-np.dot(H, self.beta))
        else:
            self.invCY = solve_chol(self.RC, Y)
            self.invCY = self.invCY[:, np.newaxis]
            self.beta = np.array([])
            self.RHCH = np.array([])

    def posterior(self, K12, DK22, H1=np.array([]), H2=np.array([])):
        if H1:
            self.mu = np.dot(H2, self.beta) + np.dot(K12.T, self.invCY)
        else:
            self.mu = np.dot(K12.T, self.invCY)

        Vf = np.linalg.solve(self.RC.T, K12)
        Covf = DK22 - np.sum(np.multiply(Vf, Vf), 0).T

        if H1:
            R2 = H2.T - np.dot(H1.T, solve_chol(self.RC, K12))
            LHCH = approx_chol(np.dot(H1.T, solve_chol(self.RC, H1)))
            Vb = np.linalg.solve(LHCH.T, R2)
            Covb = np.sum(np.multiply(Vb, Vb), 0).T
            self.var = Covb + Covf
        else:
            self.var = Covf

    def downdate(self, K, Y, i, H=np.array([])):
        n = K.shape[0]
        T = np.concatenate((np.arange(i-1), np.arange(i, n)))
        Y1 = Y[T]

        K2i = K[T, i-1]
        Kii = K[i-1, i-1]

        RC = self.RC.copy()
        #print(RC)
        RC11 = RC[0:(i-1), 0:(i-1)]
        #print(RC11)
        RC13 = RC[0:(i-1), i:n]
        #print(RC13)
        S23 = RC[i-1, i:n]
        #print(S23)
        S33 = RC[i:n, i:n]
        #print(S33)
        choldate.cholupdate(S33, S23.T)
        #print(S33)
        RC31 = np.zeros((RC13.T.shape[0], RC13.T.shape[1]))
        #print(RC31)
        RC1 = np.concatenate((RC11, RC13), 1)
        RC3 = np.concatenate((RC31, S33), 1)
        RC = np.concatenate((RC1, RC3), 0)

        if H:
            H1 = H[T,]
            Hi = H[i,]

            RHCH = approx_chol(np.dot(H1.T, solve_chol(RC, H1)))
            Ri = Hi - np.dot(H1.T, solve_chol(self.RC, H1))
            beta = solve_chol(RHCH, np.dot(H1.T, solve_chol(RC, Y1)))
            invCY = solve_chol(RC, Y1-np.dot(H1, beta))
            mu = np.dot(Hi.T, beta) + np.dot(K2i.T, invCY)
        else:
            invCY = solve_chol(RC, Y1)
            mu = np.dot(K2i.T, invCY)
            beta = np.array([])

        Vf = np.linalg.solve(RC.T, K2i)
        Covf = Kii - np.sum(np.multiply(Vf, Vf), 0).T

        if H:
            Vb = np.linalg.solve(RHCH.T, Ri)
            Covb = np.sum(np.multiply(Vb, Vb), 0).T
            var = Covb + Covf
        else:
            var = Covf

        return mu, var

    def likelihood(self, K, Y, H=np.array([])):
        n = Y.shape[0]
        m = np.linalg.matrix_rank(H)

        if m:
            RK = approx_chol(K)
            A = np.dot(H.T, solve_chol(RK, H))
            RA = approx_chol(A)
            KHAHK = solve_chol(RK, np.dot(H, solve_chol(RA, np.dot(H.T, solve_chol(RK, np.eye(n))))))
            self.nll = .5*np.dot(Y.T, self.invCY) - .5*np.dot(Y.T, np.dot(KHAHK, Y)) + np.sum(np.log(np.diag(RK))) + np.sum(np.log(np.diag(A))) + .5*(n-m)*np.log(2*np.pi)
        else:
            self.nll = .5*np.dot(Y.T, self.invCY) + .5*np.sum(np.log(np.diag(self.RC))) + .5*n*np.log(2*np.pi)

    def pseudo_likelihood(self, K, Y, H=np.array([])):
        n = K.shape[0]
        nll = 0.

        for i in range(1, n+1):
            mu, var = self.downdate(K, Y, i, H)
            nll += .5*np.log(var) + (Y[i-1]-mu)**2/(2*var) + .5*np.log(2*np.pi)

        self.nll = nll
