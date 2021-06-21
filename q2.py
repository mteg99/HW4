from oct2py import octave as oc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.svm import SVC
import numpy as np
import multiprocessing as mp

import q2_api as q2

N_TRAIN = 1000
N_TEST = 10000


def cross_validate(x_train, y_train, c, gamma, i, avg_pr_error):
    avg_pr_error[i] = (q2.cross_validate(x_train, y_train, 10, c, gamma))
    print('C = ' + str(c) + ', sigma = ' + str(gamma))


def main():
    # Generate data
    oc.addpath('C:/Users/matth/Documents/Machine Learning/HW4')
    oc.eval('pkg load statistics')
    x_train, y_train = oc.generateMultiringDataset(2, N_TRAIN, nout=2)
    x_test, y_test = oc.generateMultiringDataset(2, N_TEST, nout=2)
    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.reshape(N_TRAIN)
    y_test = y_test.reshape(N_TEST)

    # Plot test set
    plt.scatter(x_test[:, 0], x_test[:, 1], color=['C1' if i == 1 else 'C0' for i in y_test])
    plt.title('10000 Test Samples')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='C1', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Class 2', markerfacecolor='C0', markersize=10)])
    plt.show()

    # Cross-validate hyperparameters
    workers = []
    manager = mp.Manager()
    order = 3
    grid_size = 2 * order + 1
    avg_pr_error = manager.list(range(grid_size ** 2))
    c_scale = np.logspace(-order, order, num=grid_size)
    gamma_scale = np.logspace(-order, order, num=grid_size)
    i = 0
    for c in c_scale:
        for gamma in gamma_scale:
            w = mp.Process(target=cross_validate, args=(x_train, y_train, c, gamma, i, avg_pr_error))
            workers.append(w)
            w.start()
            i += 1
    for w in workers:
        w.join()
    avg_pr_error = np.array(avg_pr_error).reshape((grid_size, grid_size))
    c, gamma = np.unravel_index(avg_pr_error.argmin(), avg_pr_error.shape)
    c = c_scale[c]
    gamma = gamma_scale[gamma]
    print('Optimal C:')
    print(c)
    print('Optimal Sigma:')
    print(gamma)

    # Plot cross-validation heatmap
    fig, ax = plt.subplots()
    ax.imshow(avg_pr_error)
    ax.set_xticks(np.arange(len(c_scale)))
    ax.set_yticks(np.arange(len(gamma_scale)))
    ax.set_xticklabels(c_scale)
    ax.set_yticklabels(gamma_scale)
    for c in range(grid_size):
        for sigma in range(grid_size):
            text = ax.text(sigma, c, round(avg_pr_error[c, sigma], 3), ha="center", va="center", color="w")
    ax.set_title('Pr(error) Heatmap')
    ax.set_xlabel('Sigma')
    ax.set_ylabel('C').set_rotation(0)
    plt.show()

    # Train SVM with optimal hyperparameters
    svm = SVC(C=c, gamma=gamma, kernel='rbf').fit(x_train, y_train)
    y_hat = svm.predict(x_test)
    y_hat1 = y_hat[[i == 1 for i in y_test]]
    y_hat2 = y_hat[[i == 2 for i in y_test]]
    pr_error = (np.count_nonzero([i != 1 for i in y_hat1]) + np.count_nonzero([i != 2 for i in y_hat2])) / N_TEST
    print('Pr(error):')
    print(pr_error)

    # Plot classifications
    x1 = x_test[[i == 1 for i in y_test]]
    x2 = x_test[[i == 2 for i in y_test]]
    colors1 = ['green' if y_hat1[i] == 1 else 'red' for i in range(len(x1))]
    colors2 = ['green' if y_hat2[i] == 2 else 'red' for i in range(len(x2))]
    plt.scatter(x2[:, 0], x2[:, 1], marker='^', color=colors2)
    plt.scatter(x1[:, 0], x1[:, 1], marker='o', color=colors1)
    plt.title('SVM Classifications')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Class 1 Hit', markerfacecolor='g', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='Class 1 Miss', markerfacecolor='r', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='Class 2 Hit', markerfacecolor='g', markersize=10),
                        Line2D([0], [0], marker='^', color='w', label='Class 2 Mass', markerfacecolor='r', markersize=10)])
    plt.show()


if __name__ == "__main__":
    main()
