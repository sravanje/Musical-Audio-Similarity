"""
Display DTW computation from chromagrams of 2 provided files.
Custom similarity score between dtw path and baseline.

- Dhawal Modi & Sravan Jayati
"""

import argparse
import numpy as np
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
from dtaidistance import dtw, dtw_visualisation as dtwvis
from chromagram import fetch_chromagram


def dtw_cost_table(x, y, distance=None):
    if distance is None:
        distance = euclidean
    nx = len(x)
    ny = len(y)
    table = np.zeros((nx + 1, ny + 1))
    # Compute left column separately, i.e. j=0.
    table[1:, 0] = np.inf
    # Compute top row separately, i.e. i=0.
    table[0, 1:] = np.inf
    # Fill in the rest.
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            d = distance(x[i - 1], y[j - 1])
            table[i, j] = d + min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
    return table


def dtw_best_path(x, y, table):
    i = len(x)
    j = len(y)
    path = [(i, j)]
    while i > 0 or j > 0:
        minval = np.inf
        if table[i - 1][j - 1] < minval:
            minval = table[i - 1, j - 1]
            step = (i - 1, j - 1)
        if table[i - 1, j] < minval:
            minval = table[i - 1, j]
            step = (i - 1, j)
        if table[i][j - 1] < minval:
            minval = table[i, j - 1]
            step = (i, j - 1)
        path.insert(0, step)
        i, j = step
    return np.array(path)


def display_dtw_path(D, path):
    plt.imshow(D)
    plt.plot(path[:, 1], path[:, 0], 'r', label='computed distance')
    plt.plot(path[:, 1], path[:, 1], '-.', label='baseline distance')
    plt.legend()
    plt.show()


def similarity_score(D, dtw_path):
    """
    (summation of distance between each point on dtw_path to line y = x)/(total samples)
    subject to change
    """

    D = np.delete(D, 0, axis=0)
    D = np.delete(D, 0, axis=1)

    min_D = D.min()
    max_D = D.max()
    print(min_D)
    print(max_D)

    sum = 0
    z = 0
    for i in range(len(dtw_path)):
        x, y = dtw_path[i]
        z = D[x - 1, y - 1]
        sum += z

    avg_cost = sum / len(dtw_path)
    print("Average cost: ", avg_cost)

    similarity = abs(1 - ((avg_cost - min_D) / (max_D - min_D)))
    print("Normalized Similarity score: ", similarity)

    return similarity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-f1', '--file1',
                        type=str,
                        help='path to file 1')
    parser.add_argument('-f2', '--file2',
                        type=str,
                        help='path to file 2')

    args = parser.parse_args()

    file1path = args.file1
    file2path = args.file2

    # provide lower sampling rate for faster dtw
    # both chromagrams must have same length, provide end_time
    sr = 22050
    chromagram1 = fetch_chromagram(file1path, sr=sr, end_time=30)
    chromagram2 = fetch_chromagram(file2path, sr=sr, end_time=30)

    D = dtw_cost_table(chromagram1.T, chromagram2.T)
    dtw_path = dtw_best_path(chromagram1.T, chromagram2.T, D)

    dist = similarity_score(D, dtw_path)
    display_dtw_path(D, dtw_path)

    print("Score: ", dist)
