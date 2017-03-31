import numpy as np
import acquisition
import basis
import gp
import utils_bo

A = np.array([[1, 2, 3], [1, 2, 3]])
B = np.array([[1, 2, 3], [1, 2, 3]])
scales = np.array([1, 2])
#ard = np.diag(1./scales)
#C = np.dot(ard, A)
#D = np.dot(ard, B)
#print(C)
#print(D)

#print(basis.sq_dist(C, D))
#print(basis.kernel_se_norm(A, B))
#print(basis.kernel_matern(A, B, 1, scales, 5))

#R = [[1, 2, 3], [0, 4, 5], [0, 0, 6]]
R = np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]])
#Y = [7, 8, 9]
#R = np.array(R)
#Y = np.array(Y)

#print(utils_bo.cummax(R))

#print(gp.solve_chol(R, Y))
#print(gp.approx_chol(np.array([[1, 2], [1, 0]])))

H = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]

X = np.array([[1, 2, 3], [1, 2, 3]])
#print(X.shape)
K = basis.kernel_se_norm(X, X)
#K = basis.kernel_se(X, 'diag', 0.01, 0.5)
#print(K)
Y = np.array([3, 4, 5]).T
#print(Y.shape)
gp1 = gp.GP(0.01)
gp1.inference(K, Y)
#print(gp1.RC)
#print(gp1.invCY)

X2 = np.array([[1, 2], [1, 2]])
K12 = basis.kernel_se_norm(X, X2)
K22 = basis.kernel_se_norm(X2, X2)
#Y2 = np.array([3, 4, 5, 6, 7]).T
#print(K12.shape)
#print(K22.shape)
#print(Y2.shape)
#gp1.inference_update(K12, K22, Y2)
#print(gp1.RC)
#print(gp1.invCY)

#gp1.likelihood(K, Y)
#print(gp1.nll)

#DK22 = np.diag(K22)
#gp1.posterior(K12, DK22)
#print(gp1.mu)
#print(gp1.var)
#mu, var = gp1.downdate(K, Y, 3)
#print(mu)
#print(var)

gp1.pseudo_likelihood(K, Y)
#print(gp1.nll)

f, Xs, Ys, Xt, Yt, Kss = utils_bo.sample(plot=False)

utils_bo.bo(f, Xt, Yt, Xs, 100, plot=False)
