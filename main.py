import os
import time
import numpy as np
import math
import pylab as pl
import pickle
import matplotlib.pyplot as plt

from utils import evaluateDataset, plotRes, build
from clfs import *

import target
import ho.utils_oo as utils_oo

# Macros
HORIZON = 1
RHOMAX = 20
SIGMA = 0.1
DELTA = 0.05
EPOCH = 1
UPDATE = False
VERBOSE = True
PARALLEL = False
JOBS = 8
NSPLITS = 3
SAVE = True
PATH = "data/"
PLOT = False
MODEL = 'SVM'

alpha_ = math.log(HORIZON) * (SIGMA ** 2)
#rhos_hoo = [float(j)/float(RHOMAX-1) for j in range(RHOMAX)]
rhos_hoo = [0.9]
nu_ = 1.

if __name__ == '__main__':
    start_time = time.time()

    #models = [SVM(), MLP(), GBM(), KNN()]
    #params = [d_svm, d_mlp, d_gbm, d_knn]
    if MODEL == 'SVM':
        models = [SVM()]
        params = [d_svm]
    elif MODEL == 'GBM':
        models = [GBM()]
        params = [d_gbm]
    elif MODEL == 'KNN':
        models = [KNN()]
        params = [d_knn]
    elif MODEL == 'MLP':
        models = [MLP()]
        params = [d_mlp]

    path = os.path.join(os.getcwd(), 'datasets')
    #datasets = ['aff.csv', 'pinter.csv', 'breast_cancer.csv', 'indian_liver.csv', 'parkinsons.csv',
    #            'lsvt.csv', 'pima-indians-diabetes.csv']
    datasets = ['wine.csv']
    #problems = ['cont', 'binary', 'binary', 'binary', 'binary', 'binary', 'binary']
    problems = ['cont']
    #targets = [0, 0, 0, 10, 16, 0, 8]
    targets = [0]

    for model, parameter_dict in zip(models, params):
        print('Evaluating model {}'.format(model.name))
        for dataset, tg, problem in zip(datasets, targets, problems):
            model.problem = problem
            #np.random.seed(93)
            #print(np.random.randn(1))
            try:
                g, g2, r = evaluateDataset(os.path.join(path, dataset), target_index=tg, model=model,
                                               parameter_dict=parameter_dict, method='5fold', seed=20,
                                               max_iter=HORIZON, problem=problem)
                if SAVE and VERBOSE:
                    print("Writing BO data...")
                    #with open("data/PI", 'wb') as file:
                    #    pickle.dump(g4, file)
                    with open("data/EI", 'wb') as file:
                        pickle.dump(g, file)
                    with open("data/UCB1", 'wb') as file:
                        pickle.dump(g2, file)
                    #with open("data/UCB2", 'wb') as file:
                    #    pickle.dump(g5, file)
                    with open("data/R", 'wb') as file:
                        pickle.dump(r, file)

                if PLOT:
                    plotRes(g, g2, r, dataset, model, problem=problem)
                #print(np.random.randn(1))
            except Exception as e:
                print(e)
                continue

    path = os.path.join(os.getcwd(), 'datasets')
    model = models[0]
    param = params[0]
    dataset = datasets[0]
    problem = problems[0]
    target_index = targets[0]
    X, y = build(os.path.join(path, dataset), target_index)

    if MODEL == 'SVM':
        f_hyper = target.LossSVM(model, X, y, '5fold', problem)
        bbox = utils_oo.std_box(f_hyper.f, None, NSPLITS, SIGMA, [param['C'][1], param['gamma'][1]], [param['C'][0], param['gamma'][0]])
    elif MODEL == 'GBM':
        f_hyper = target.LossGBM(model, X, y, '5fold', problem)
        bbox = utils_oo.std_box(f_hyper.f, None, NSPLITS, SIGMA, [param['learning_rate'][1], param['n_estimators'][1], param['max_depth'][1], param['min_samples_split'][1]], [param['learning_rate'][0], param['n_estimators'][0], param['max_depth'][0], param['min_samples_split'][0]])
    elif MODEL == 'KNN':
        f_hyper = target.LossKNN(model, X, y, '5fold', problem)
        bbox = utils_oo.std_box(f_hyper.f, None, NSPLITS, SIGMA, [param['n_neighbors'][1]], [param['n_neighbors'][0]])
    elif MODEL == 'MLP':
        f_hyper = target.LossMLP(model, X, y, '5fold', problem)
        bbox = utils_oo.std_box(f_hyper.f, None, NSPLITS, SIGMA, [param['hidden_layer_size'][1], param['alpha'][1]], [param['hidden_layer_size'][0], param['alpha'][0]])

    data = [None for k in range(EPOCH)]
    current = [[0. for i in range(HORIZON)] for j in range(RHOMAX)]
    #current = [0. for i in range(HORIZON)]
    #rhos_poo = utils_oo.get_rhos(NSPLITS, 0.9, HORIZON)

    # HOO
    if VERBOSE:
        print("HOO!")

    if PARALLEL:
        for k in range(EPOCH):
            data[k] = pool.map(partial_regret_hoo, rhos_hoo)
            if VERBOSE:
                print(str(k+1)+"/"+str(EPOCH))
            #print(regrets.shape)
    else:
        for k in range(EPOCH):
            for j in range(len(rhos_hoo)):
                losses = utils_oo.loss_hoo(bbox, rhos_hoo[j], nu_, alpha_, SIGMA, HORIZON, UPDATE)
                for i in range(HORIZON):
                    current[j][i] += losses[i]
                if VERBOSE:
                    print(str(1+j+k*len(rhos_hoo))+"/"+str(EPOCH*len(rhos_hoo)))
            data[k] = current
            current = [[0. for i in range(HORIZON)] for j in range(len(rhos_hoo))]

    if SAVE and VERBOSE:
        print("Writing HO data...")
        for k in range(EPOCH):
            with open("data/HOO"+str(k+1), 'wb') as file:
                pickle.dump(data[k], file)
            print(str(k+1)+"/"+str(EPOCH))

    """
    if VERBOSE:
        print("POO!")

    losses = utils_oo.loss_poo(bbox, rhos_poo, nu_, alpha_, HORIZON, EPOCH)
    data_poo = losses

    if SAVE and VERBOSE:
        print("Writing HO data...")
        with open("data/POO", 'wb') as file:
            pickle.dump(data_poo, file)
    """
    print("--- %s seconds ---" % (time.time() - start_time))

    data = [None for k in range(EPOCH)]
    for k in range(EPOCH):
        with open("data/HOO"+str(k+1), 'rb') as file:
            data[k] = pickle.load(file)
    #with open("data/POO", 'rb') as file:
    #    data_poo = pickle.load(file)
    with open("data/R", 'rb') as file:
        data_random = pickle.load(file)
    #with open("data/PI", 'rb') as file:
    #    data_pi = pickle.load(file)
    with open("data/EI", 'rb') as file:
        data_ei = pickle.load(file)
    with open("data/UCB1", 'rb') as file:
        data_ucb1 = pickle.load(file)
    #with open("data/UCB2", 'rb') as file:
    #    data_ucb2 = pickle.load(file)

    #length_poo = len(rhos_poo)
    length_hoo = len(rhos_hoo)
    #rhostoshow = [int(length_hoo*k/4.) for k in range(4)]
    rhostoshow = rhos_hoo
    #style = [[5,5], [1,3], [5,3,1,3], [5,2,5,2,5,10]]

    means = [[sum([data[k][j][i] for k in range(EPOCH)])/float(EPOCH) for i in range(HORIZON)] for j in range(length_hoo)]
    #means_poo = [sum([data_poo[u][v]/float(EPOCH) for u in range(EPOCH)]) for v in range(HORIZON)]

    x = np.arange(0, HORIZON + 1)
    fig = plt.figure()
    plt.plot(x, -data_random, label='Random search', color='red')
    plt.plot(x, -data_ei, label='EI', color='blue')
    #plt.plot(x, -data_pi, label='PI', color='cyan')
    plt.plot(x, -data_ucb1, label=r'GPUCB', color='yellow')
    #plt.plot(x, -data_ucb2[1:], label=r'GPUCB ($\beta=1.5$)', color='green')
    for i in range(len(rhostoshow)):
        k = rhostoshow[i]
        label__ = r"$\mathtt{HOO}$"
        array = means[i]
        array.insert(0, means[i][0])
        plt.plot(x, -np.array(array), label=label__, color ='green')
    #plt.plot(x, -np.array(means_poo), label=r"$\mathtt{POO}$")

    plt.grid()
    plt.legend(loc=0)
    plt.xlabel('Number of evaluations')
    if problem == 'binary':
        plt.ylabel('Log-loss')
    else:
        plt.ylabel('MSE')
    datasetname = datasets[0].split('.')[0]
    plt.savefig(os.path.join(os.path.abspath('.'), 'results/{}/{}.pdf'.format(model.name, datasetname)))
    plt.close(fig)
