from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def evaluate_MLP(x_train, y_train, x_test, y_test, p):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(p, activation=tf.nn.softplus))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='SGD', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=100, verbose=0)
    loss = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
    return loss


def cross_validate(x, y, k, p):
    loss = []
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(x):
        x_train = np.array([x[i] for i in train_index])
        y_train = np.array([y[i] for i in train_index])
        x_test = np.array([x[i] for i in test_index])
        y_test = np.array([y[i] for i in test_index])
        loss.append(evaluate_MLP(x_train, y_train, x_test, y_test, p))
    avg_loss = np.sum(loss) / k
    return avg_loss
