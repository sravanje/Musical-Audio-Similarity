"""
Display DTW computation from chromagrams of 2 provided files.
Custom similarity score between dtw path and baseline.

- Dhawal Modi & Sravan Jayati
"""

import argparse
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from chromagram import fetch_chromagram

import os
from tqdm import tqdm


def dtw_table(x, y, distance=None):

    # https://musicinformationretrieval.com/dtw_example.html

    if distance is None:
        distance = euclidean
    nx = len(x)
    ny = len(y)
    table = np.zeros((nx+1, ny+1))
    # Compute left column separately, i.e. j=0.
    table[1:, 0] = np.inf
    # Compute top row separately, i.e. i=0.
    table[0, 1:] = np.inf
    # Fill in the rest.
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            d = distance(x[i-1], y[j-1])
            table[i, j] = d + min(table[i-1, j], table[i, j-1], table[i-1, j-1])
    return table

def dtw(x, y, table):

    # https://musicinformationretrieval.com/dtw_example.html

    i = len(x)
    j = len(y)
    path = [(i, j)]
    while i > 0 or j > 0:
        minval = np.inf
        if table[i-1][j-1] < minval:
            minval = table[i-1, j-1]
            step = (i-1, j-1)
        if table[i-1, j] < minval:
            minval = table[i-1, j]
            step = (i-1, j)
        if table[i][j-1] < minval:
            minval = table[i, j-1]
            step = (i, j-1)
        path.insert(0, step)
        i, j = step
    return np.array(path)

def display_dtw_path(D, path):
    plt.imshow(D)
    plt.plot(path[:,1],path[:,0], 'r', label='computed path')
    plt.legend()
    plt.show()

def similarity_score(D, dtw_path):
    '''
    (summation of distance between each point on dtw_path to line y = x)/(total samples)
    subject to change
    '''
    sum=0
    for i in dtw_path:
        sum += D[i[0],i[1]]

    return sum/len(dtw_path)

if __name__=="__main__":
    
    # parser = argparse.ArgumentParser()

    # parser.add_argument('-f1', '--file1', 
    #                     type = str,  
    #                     help ='path to file 1')
    # parser.add_argument('-f2', '--file2', 
    #                     type = str,  
    #                     help ='path to file 2')
    
    # args = parser.parse_args()

    # file1path = args.file1
    # file2path = args.file2

    # # provide lower sampling rate for faster dtw
    # # both chromagrams must have same length, provide end_time
    # sr = 10000
    # chromagram1 = fetch_chromagram(file1path, sr=sr, end_time=10)
    # chromagram2 = fetch_chromagram(file2path, sr=sr, end_time=10)

    # D = dtw_table(chromagram1.T, chromagram2.T)
    # dtw_path = dtw(chromagram1.T, chromagram1.T, D)

    # display_dtw_path(D, dtw_path)

    # print("similarity score between provided files: ", similarity_score(D, dtw_path))

    # TEST SESSION COMPARING SIMILAR AND DISSIMILAR AUDIO FILES

    file_pair_1 = ['../tests/1727_tempo_matched.wav',
                    '../tests/1727_schubert_op114_2.wav',
                    '../tests/1727_schubert_op114_2_crowd.wav',
                    '../tests/1727_schubert_op114_2_whitenoise.wav',]
    file_pair_2 = ['../tests/1733_tempo_matched.wav',
                    '../tests/1733_tempo_matched_offset.wav',
                    '../tests/1733_sy_sps92.wav']

    # similar_pairs:
    sim_scores = {}
    for i in tqdm(range(len(file_pair_1))):
        for j in range(i,len(file_pair_1)):
            if j==i:
                continue
            file1path = file_pair_1[i]
            file2path = file_pair_1[j]

            # provide lower sampling rate for faster dtw
            # both chromagrams must have same length, provide end_time
            sr = 10000
            chromagram1 = fetch_chromagram(file1path, sr=sr, end_time=10)
            chromagram2 = fetch_chromagram(file2path, sr=sr, end_time=10)

            D = dtw_table(chromagram1.T, chromagram2.T)
            dtw_path = dtw(chromagram1.T, chromagram1.T, D)

            # display_dtw_path(D, dtw_path)

            sim_scores[(os.path.basename(file1path),os.path.basename(file2path))] = similarity_score(D, dtw_path)

    for i in tqdm(range(len(file_pair_2))):
        for j in range(i,len(file_pair_2)):
            if j==i:
                continue
            file1path = file_pair_2[i]
            file2path = file_pair_2[j]

            # provide lower sampling rate for faster dtw
            # both chromagrams must have same length, provide end_time
            sr = 10000
            chromagram1 = fetch_chromagram(file1path, sr=sr, end_time=10)
            chromagram2 = fetch_chromagram(file2path, sr=sr, end_time=10)

            D = dtw_table(chromagram1.T, chromagram2.T)
            dtw_path = dtw(chromagram1.T, chromagram1.T, D)

            # display_dtw_path(D, dtw_path)

            sim_scores[(os.path.basename(file1path),os.path.basename(file2path))] = similarity_score(D, dtw_path)
    
    # ------------
    # dissimilar_pairs:
    dsim_scores = {}
    for i in tqdm(range(len(file_pair_1))):
        for j in range(len(file_pair_2)):
            file1path = file_pair_1[i]
            file2path = file_pair_2[j]

            # provide lower sampling rate for faster dtw
            # both chromagrams must have same length, provide end_time
            sr = 10000
            chromagram1 = fetch_chromagram(file1path, sr=sr, end_time=10)
            chromagram2 = fetch_chromagram(file2path, sr=sr, end_time=10)

            D = dtw_table(chromagram1.T, chromagram2.T)
            dtw_path = dtw(chromagram1.T, chromagram1.T, D)

            # display_dtw_path(D, dtw_path)

            dsim_scores[(os.path.basename(file1path),os.path.basename(file2path))] = similarity_score(D, dtw_path)
    
    a = list(sim_scores.values())
    b = list(dsim_scores.values())
    print("sim scores: ", a, '\n')
    print("dsim scores: ", b, '\n')

    import matplotlib.pyplot as plt

    x_labels_a = [f'a{i}' for i in range(len(a))]  # Labels for list a
    x_labels_b = [f'b{i}' for i in range(len(b))]  # Labels for list b

    # Create the bar plots
    plt.bar(x_labels_a, a, label='similar pairs', color='blue')
    plt.bar(x_labels_b, b, label='dissimilar pairs', color='red')

    # Add labels and legend
    plt.ylabel('similarity cost')
    plt.title('Comparison of similar and dissimilar pairs')
    plt.legend()

    # Show the plot
    plt.show()

    data = [a, b]

    # Create labels for the x-axis
    labels = ['similar pairs', 'dissimilar pairs']

    # Create the side-by-side vertical box plots
    plt.boxplot(data, labels=labels)

    # Add labels and title
    plt.ylabel('similarity cost')
    plt.title('Side-by-Side Vertical Box Plots of similar and dissimilar pairs')

    # Show the plot
    plt.show()
