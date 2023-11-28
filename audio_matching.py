import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt

sys.path.append('..')
import libfmp.b
import libfmp.c4
import libfmp.c7
#%matplotlib inline


def compute_cens_from_file(fn_wav, Fs=22050, N=4410, H=2205, ell=21, d=5):
    """Compute CENS features from file

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        fn_wav (str): Filename of wav file
        Fs (scalar): Feature rate of wav file (Default value = 22050)
        N (int): Window size for STFT (Default value = 4410)
        H (int): Hop size for STFT (Default value = 2205)
        ell (int): Smoothing length (Default value = 21)
        d (int): Downsampling factor (Default value = 5)

    Returns:
        X_CENS (np.ndarray): CENS features
        L (int): Length of CENS feature sequence
        Fs_CENS (scalar): Feature rate of CENS features
        x_duration (float): Duration (seconds) of wav file
    """
    x, Fs = librosa.load(fn_wav, sr=Fs)
    x_duration = x.shape[0] / Fs
    X_chroma = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    X_CENS, Fs_CENS = libfmp.c7.compute_cens_from_chromagram(X_chroma, Fs=Fs / H, ell=ell, d=d)
    L = X_CENS.shape[1]
    return X_CENS, L, Fs_CENS, x_duration


def compute_matching_function_dtw(X, Y, stepsize=2):
    """Compute CENS features from file

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        X (np.ndarray): Query feature sequence (given as K x N matrix)
        Y (np.ndarray): Database feature sequence (given as K x M matrix)
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        Delta (np.ndarray): DTW-based matching function
        C (np.ndarray): Cost matrix
        D (np.ndarray): Accumulated cost matrix
    """
    C = libfmp.c7.cost_matrix_dot(X, Y)
    if stepsize == 1:
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw(C)
    if stepsize == 2:
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    N, M = C.shape
    Delta = D[-1, :] / N
    return Delta, C, D


def matches_dtw(pos, D, stepsize=2):
    """Derives matches from positions for DTW-based strategy

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        pos (np.ndarray): End positions of matches
        D (np.ndarray): Accumulated cost matrix
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        matches (np.ndarray): Array containing matches (start, end)
    """
    matches = np.zeros((len(pos), 2)).astype(int)
    for k in range(len(pos)):
        t = pos[k]
        matches[k, 1] = t
        if stepsize == 1:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw(D, m=t)
        if stepsize == 2:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw_21(D, m=t)
        s = P[0, 1]
        matches[k, 0] = s
    return matches


def compute_plot_matching_function_DTW(fn_wav_X, fn_wav_Y, fn_ann,ell=21, d=5, stepsize=2, tau=0.2, num=5, ylim=[0, 0.35]):
    ann, _ = libfmp.c4.read_structure_annotation(fn_ann)
    color_ann = {'Theme': [0, 0, 1, 0.1], 'Match': [0, 0, 1, 0.2]}
    X, N, Fs_X, x_duration = compute_cens_from_file(fn_wav_X, ell=ell, d=d)
    Y, M, Fs_Y, y_duration = compute_cens_from_file(fn_wav_Y, ell=ell, d=d)
    Delta, C, D = compute_matching_function_dtw(X, Y, stepsize=stepsize)
    pos = libfmp.c7.mininma_from_matching_function(Delta, rho=2 * N // 3, tau=tau, num=num)
    matches = matches_dtw(pos, D, stepsize=stepsize)

    fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1],
                                              'height_ratios': [1, 1]}, figsize=(8, 4))
    cmap = libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)
    libfmp.b.plot_matrix(C, Fs=Fs_X, ax=[ax[0]], ylabel='Time (seconds)',
                         title='Cost matrix $C$ with ground truth annotations (blue rectangles)',
                         colorbar=False, cmap=cmap)
    libfmp.b.plot_segments_overlay(ann, ax=ax[0], alpha=0.2, time_max=y_duration,
                                   colors=color_ann, print_labels=False)

    title = r'Matching function $\Delta_\mathrm{DTW}$ with matches (red rectangles)'
    libfmp.b.plot_signal(Delta, ax=ax[1], Fs=Fs_X, color='k', title=title, ylim=ylim)
    ax[1].grid()
    libfmp.c7.plot_matches(ax[1], matches, Delta, Fs=Fs_X, s_marker='', t_marker='o')
    plt.tight_layout()
    plt.show()


data_dir = os.path.join('fmp', 'data', 'C7')
fn_wav_all = [os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Bernstein.wav'),
              os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Karajan.wav'),
              os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Scherbakov.wav')]
fn_ann_all = [os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Bernstein_Theme.csv'),
              os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Karajan_Theme.csv'),
              os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Scherbakov_Theme.csv')]
names_all = ['Bernstein', 'Karajan', 'Scherbakov (piano version)']
fn_wav_X = os.path.join(data_dir, 'FMP_C7_Audio_Beethoven_Op067-01_Bernstein_Theme_1.wav')

for f in range(3):
    print('=== Query X: Bernstein (Theme 1); Database Y:', names_all[f], ' ===')
    compute_plot_matching_function_DTW(fn_wav_X, fn_wav_all[f], fn_ann_all[f])