from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import numpy as np
from oct2py import octave as oc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import multiprocessing as mp

import q1_api as q1

N_TRAIN = 1000
N_TEST = 10000


def cross_validate_p(x_train, y_train, p, avg_loss):
    avg_loss.append(q1.cross_validate(x_train, y_train, k=10, p=p))
    print('P = ' + str(p))


def main():
    # Generate data
    oc.addpath('C:/Users/matth/Documents/Machine Learning/HW4')
    oc.eval('pkg load statistics')
    x_train, y_train = oc.exam4q1_generateData(N_TRAIN, nout=2)
    x_test, y_test = oc.exam4q1_generateData(N_TEST, nout=2)
    x_train = x_train.reshape(N_TRAIN)
    y_train = y_train.reshape(N_TRAIN)
    x_test = x_test.reshape(N_TEST)
    y_test = y_test.reshape(N_TEST)

    # Plot test set
    plt.scatter(x_test, y_test)
    plt.title('10000 Test Samples')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

    # Cross-validate number of perceptrons
    workers = []
    manager = mp.Manager()
    avg_loss = manager.list()
    for p in range(1, 17):
        w = mp.Process(target=cross_validate_p, args=(x_train, y_train, p, avg_loss))
        workers.append(w)
        w.start()
    for w in workers:
        w.join()
    p = np.argmin(avg_loss) + 1
    print('Optimal P:')
    print(p)

    # Plot MSE vs number of perceptrons
    plt.plot(range(1, 17), avg_loss)
    plt.title('Average MSE vs Perceptrons')
    plt.xlabel('Perceptrons')
    plt.ylabel('Average MSE')
    plt.show()

    # Train MLP with optimal number of perceptrons
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(p, activation=tf.nn.softplus))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='SGD', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=1000)
    y_hat = model.predict(x_test, batch_size=32)
    mse = model.evaluate(x_test, y_test)
    print('MSE:')
    print(mse)

    # Plot predictions
    plt.scatter(x_test, y_test)
    plt.scatter(x_test, y_hat)
    plt.title('MLP Predictions (P = ' + str(p) + ', MSE = ' + str(round(mse, 3)) + ')')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(handles=[Line2D([0], [0], marker='o', color='w', label='Test Set', markerfacecolor='C0', markersize=10),
                        Line2D([0], [0], marker='o', color='w', label='MLP Predictions', markerfacecolor='C1', markersize=10)])
    plt.show()


if __name__ == "__main__":
    main()
