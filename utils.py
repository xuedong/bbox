import os

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

from bo.bo import BO
from bo.surrogates.GaussianProcess import GaussianProcess
from bo.acquisition import Acquisition
from bo.covfunc import squaredExponential

import ho.hoo as hoo
import ho.poo as poo
import ho.utils_oo as utils_oo
import target

class loss:
    def __init__(self, model, X, y, method='holdout', problem='binary'):
        self.model = model
        self.X = X
        self.y = y
        self.method = method
        self.problem = problem
        sc = StandardScaler()
        self.X = sc.fit_transform(self.X)
        if self.problem == 'binary':
            self.loss = log_loss
        elif self.problem == 'cont':
            self.loss = mean_squared_error
        else:
            self.loss = log_loss

    def evaluateLoss(self, **param):
        if self.method == 'holdout':
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=93)
            clf = self.model.__class__(**param, problem=self.problem).eval()
            clf.fit(X_train, y_train)
            if self.problem == 'binary':
                yhat = clf.predict_proba(X_test)[:, 1]
            elif self.problem == 'cont':
                yhat = clf.predict(X_test)
            else:
                yhat = clf.predict_proba(X_test)
            return (- self.loss(y_test, yhat))
        elif self.method == '5fold':
            kf = KFold(n_splits=5, shuffle=False)
            losses = []
            for train_index, test_index in kf.split(self.X):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]
                clf = self.model.__class__(**param, problem=self.problem).eval()
                clf.fit(X_train, y_train)
                if self.problem == 'binary':
                    yhat = clf.predict_proba(X_test)[:, 1]
                elif self.problem == 'cont':
                    yhat = clf.predict(X_test)
                else:
                    yhat = clf.predict_proba(X_test)
                losses.append(- self.loss(y_test, yhat))
            return (np.average(losses))


def cumMax(history):
    n = len(history)
    res = np.empty((n,))
    for i in range(n):
        res[i] = np.max(history[:(i + 1)])
    return (res)


def build(csv_path, target_index, header=None):
    data = pd.read_csv(csv_path, header=header)
    data = data.as_matrix()
    y = data[:, target_index]
    X = np.delete(data, obj=np.array([target_index]), axis=1)
    return X, y


def evaluateDataset(csv_path, target_index, problem, model, parameter_dict, method='holdout', seed=20, max_iter=50):
    print('Now evaluating {}...'.format(csv_path))
    X, y = build(csv_path, target_index)

    wrapper = loss(model, X, y, method=method, problem=problem)
    """
    print('Evluating PI')
    np.random.seed(seed)
    sexp = squaredExponential()
    gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    acq_pi = Acquisition(mode='ProbabilityImprovement')
    gpgo_pi = BO(gp, acq_pi, wrapper.evaluateLoss, parameter_dict, n_jobs=1)
    gpgo_pi.run(max_iter=max_iter)
    """
    print('Evaluating EI')
    np.random.seed(seed)
    sexp = squaredExponential()
    gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    acq_ei = Acquisition(mode='ExpectedImprovement')
    gpgo_ei = BO(gp, acq_ei, wrapper.evaluateLoss, parameter_dict, n_jobs=1)
    gpgo_ei.run(max_iter=max_iter)

    # Also add UCB, beta = 0.5, beta = 1.5
    print('Evaluating GP-UCB beta = 0.5')
    np.random.seed(seed)
    sexp = squaredExponential()
    gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    acq_ucb = Acquisition(mode='UCB', beta=0.5)
    gpgo_ucb = BO(gp, acq_ucb, wrapper.evaluateLoss, parameter_dict, n_jobs=1)
    gpgo_ucb.run(max_iter=max_iter)

    #print('Evaluating GP-UCB beta = 1.5')
    #np.random.seed(seed)
    #sexp = squaredExponential()
    #gp = GaussianProcess(sexp, optimize=True, usegrads=True)
    #acq_ucb2 = Acquisition(mode='UCB', beta=1.5)
    #gpgo_ucb2 = BO(gp, acq_ucb2, wrapper.evaluateLoss, parameter_dict, n_jobs=1)
    #gpgo_ucb2.run(max_iter=max_iter)

    print('Evaluating random')
    np.random.seed(seed)
    r = evaluateRandom(gpgo_ei, wrapper.evaluateLoss, n_eval=max_iter + 1)
    r = cumMax(r)

    #pi_h = np.array(gpgo_pi.history)
    ei_h = np.array(gpgo_ei.history)
    ucb1_h = np.array(gpgo_ucb.history)
    #ucb2_h = np.array(gpgo_ucb2.history)

    return ei_h, ucb1_h, r


def plotRes(gpgoei_history, gpgoucb_history, random, datasetname, model, problem):
    import matplotlib
    import matplotlib.pyplot as plt
    x = np.arange(1, len(random) + 1)
    fig = plt.figure()
    plt.plot(x, -random, label='Random search', color='red')
    plt.plot(x, -gpgoei_history, label='EI', color='blue')
    #plt.plot(x, -gpgopi_history, label='PI', color='cyan')
    plt.plot(x, -gpgoucb_history, label=r'GPUCB ($\beta=.5$)', color='yellow')
    #plt.plot(x, -gpgoucb2_history, label=r'GPUCB ($\beta=1.5$)', color='green')
    plt.grid()
    plt.legend(loc=0)
    plt.xlabel('Number of evaluations')
    if problem == 'binary':
        plt.ylabel('Log-loss')
    else:
        plt.ylabel('MSE')
    datasetname = datasetname.split('.')[0]
    plt.savefig(os.path.join(os.path.abspath('.'), 'testing/results/{}/{}.pdf'.format(model.name, datasetname)))
    plt.close(fig)
    return None


def evaluateRandom(gpgo, loss, n_eval=20):
    res = []
    for i in range(n_eval):
        param = gpgo._sampleParam()
        l = loss(**param)
        res.append(l)
        print('Param {}, loss: {}'.format(param, l))
    return (res)
