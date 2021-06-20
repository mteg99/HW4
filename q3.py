import cv2
import numpy as np
import colorsys
import argparse
import multiprocessing as mp
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.mixture import GaussianMixture

import q3_api as q3


def cross_validate_m(x, m, avg_log_likelihood):
    avg_log_likelihood.append(q3.cross_validate(x, m, 10))
    print('M = ' + str(m))


def main():
    # Parse image filename
    parser = argparse.ArgumentParser(description='Segment image.')
    parser.add_argument('image')
    args = parser.parse_args()

    # Preprocess image
    plane = cv2.imread(args.image)
    n_rows = plane.shape[0]
    n_columns = plane.shape[1]
    n = n_rows * n_columns
    x = np.zeros(shape=(n, 5))
    for r in range(n_rows):
        for c in range(n_columns):
            i = r * n_columns + c
            x[i][0] = r
            x[i][1] = c
            x[i][2] = plane[r][c][2]
            x[i][3] = plane[r][c][1]
            x[i][4] = plane[r][c][0]
    x = minmax_scale(x, feature_range=(0, 1))

    # Cross-validate number of components
    workers = []
    manager = mp.Manager()
    avg_log_likelihood = manager.list()
    for m in range(1, 11):
        w = mp.Process(target=cross_validate_m, args=(x, m, avg_log_likelihood))
        workers.append(w)
        w.start()
    for w in workers:
        w.join()
    m = np.argmax(avg_log_likelihood) + 1
    print('Optimal M:')
    print(m)

    plt.plot(avg_log_likelihood)
    plt.show()

    # Segment image
    gmm = GaussianMixture(n_components=m).fit(x)
    segments = gmm.predict(x)
    segmented_img = np.zeros(shape=(n_rows, n_columns, 3))
    for r in range(n_rows):
        for c in range(n_columns):
            i = r * n_columns + c
            HSV = [(j * 1.0 / m, 0.5, 0.5) for j in range(m)]
            RGB = list(map(lambda j: colorsys.hsv_to_rgb(*j), HSV))
            segmented_img[r][c][0] = RGB[segments[i]][2]
            segmented_img[r][c][1] = RGB[segments[i]][1]
            segmented_img[r][c][2] = RGB[segments[i]][0]
    cv2.imshow('Image', plane)
    cv2.imshow('Segmented Image', segmented_img)
    cv2.waitKey()


if __name__ == "__main__":
    main()
