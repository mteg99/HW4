import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold


def cross_validate(x, y, k, c, gamma):
    pr_error = []
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(x):
        x_train = np.array([x[i] for i in train_index])
        y_train = np.array([y[i] for i in train_index])
        x_test = np.array([x[i] for i in test_index])
        y_test = np.array([y[i] for i in test_index])

        svm = SVC(C=c, gamma=gamma, kernel='rbf').fit(x_train, y_train)
        y_hat = svm.predict(x_test)
        y_hat1 = y_hat[[i == 1 for i in y_test]]
        y_hat2 = y_hat[[i == 2 for i in y_test]]
        pr_error = (np.count_nonzero([i != 1 for i in y_hat1]) + np.count_nonzero([i != 2 for i in y_hat2])) / len(y_hat)
    avg_pr_error = np.sum(pr_error) / k
    return avg_pr_error
