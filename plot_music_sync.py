import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import librosa


def similarity_score(D, wp):
    s = 0
    min_D = D.min()
    max_D = D.max()

    for i in range(len(wp)):
        x, y = wp[i]
        z = D[x, y]
        s += z

    avg_cost = s / len(wp)
    similarity = abs(1 - ((avg_cost - min_D) / (max_D - min_D)))

    print(avg_cost)
    print(similarity)

    return similarity


def get_dtw(chroma1, chroma2):
    if len(chroma1) < len(chroma2):
        D, wp = librosa.sequence.dtw(X=chroma1, Y=chroma2, subseq=True, metric='euclidean')
        wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)
    else:
        D, wp = librosa.sequence.dtw(X=chroma1, Y=chroma2, metric='euclidean')
        wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop_length)

    return D, wp, wp_s


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

    x_1, fs = librosa.load(file1path)
    x_2, fs = librosa.load(file2path)

    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)

    librosa.display.waveshow(x_1, sr=fs, ax=ax[0], color='blue')
    ax[0].set(title='File $X_1$')
    ax[0].label_outer()

    librosa.display.waveshow(x_2, sr=fs, ax=ax[1], color='red')
    ax[1].set(title='File $X_2$')

    hop_length = 1024

    x_1_chroma = librosa.feature.chroma_cqt(y=x_1, sr=fs, hop_length=hop_length)
    x_2_chroma = librosa.feature.chroma_cqt(y=x_2, sr=fs, hop_length=hop_length)

    fig, ax = plt.subplots(nrows=2, sharey=True)

    img = librosa.display.specshow(x_1_chroma, x_axis='time', y_axis='chroma', hop_length=hop_length, ax=ax[0])
    ax[0].set(title='Chroma Representation of $X_1$')

    librosa.display.specshow(x_2_chroma, x_axis='time', y_axis='chroma', hop_length=hop_length, ax=ax[1])
    ax[1].set(title='Chroma Representation of $X_2$')

    fig.colorbar(img, ax=ax)

    cost_matrix, warping_paths, wp_s = get_dtw(x_1_chroma, x_2_chroma)
    print("Similarity score:", similarity_score(cost_matrix, warping_paths))

    fig, ax = plt.subplots()
    img = librosa.display.specshow(cost_matrix, x_axis='time', y_axis='time', sr=fs, cmap='gray_r',
                                   hop_length=hop_length, ax=ax)
    ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
    ax.set(title='Warping Path on Acc. Cost Matrix $D$', xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
    fig.colorbar(img, ax=ax)

    plt.show()
