import numpy as np
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture


def cross_validate(x, m, k):
    log_likelihood = []
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(x):
        x_train = np.array([x[i] for i in train_index])
        x_test = np.array([x[i] for i in test_index])
        gmm = GaussianMixture(n_components=m).fit(x_train)
        log_likelihood.append(gmm.score(x_test))
    avg_log_likelihood = np.sum(log_likelihood) / k
    return avg_log_likelihood
