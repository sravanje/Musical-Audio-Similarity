import numpy as np
import librosa
import numpy as np
import tqdm
import os
import noisereduce as nr
import soundfile as sf
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

from chromagram import fetch_chromagram
from calculate_similarity import dtw_cost_table, dtw_best_path, display_dtw_path, similarity_score

gen_path = "../dataset3/archive/musicnet/custom_generated_musicnet/"
og_path = "../dataset3/archive/musicnet/musicnet/train_data/"

# where train files chromagrams will be saved, to be loaded and used during inference
chroma_path = "chromagrams/30sec_sr22050_numpy_files/"

file_pairs = []
no_pairs = []
multiple_pairs = []

og_files = os.listdir(og_path)

for file in tqdm.tqdm(os.listdir(gen_path)):
    file_num = file.split('_')[0]
    topair = [f for f in og_files if file_num in f]
    if len(topair) == 0:
        no_pairs.append(file)
        continue
    elif len(topair) > 1:
        multiple_pairs.append([file, topair])
        continue
    topair = topair[0]
    file_pairs.append([topair, file])
file_pairs = np.array(file_pairs)

# save chromagrams
og_files = file_pairs[:, 0]
for ogf in tqdm.tqdm(og_files):
    # change accordingly
    chromagram = fetch_chromagram(og_path + ogf, sr=5000)
    np.save(chroma_path + ogf[:-3] + 'npy', chromagram)


def similarity_score(D, wp):
    s = 0
    D = np.delete(D, 0, axis=0)
    D = np.delete(D, 0, axis=1)
    min_D = D.min()
    max_D = D.max()

    for i in range(len(wp)):
        x, y = wp[i]
        z = D[x - 1, y - 1]
        s += z

    avg_cost = s / len(wp)
    similarity = abs(1 - ((avg_cost - min_D) / (max_D - min_D)))

    return similarity


def method1(chromagram1, chromagram2):
    # dtw from MIR with cost with normalization
    D = dtw_cost_table(chromagram1.T, chromagram2.T)
    dtw_path = dtw_best_path(chromagram1.T, chromagram1.T, D)

    score = similarity_score(D, dtw_path)

    return score


def method2(chromagram1, chromagram2, subsequence=False):
    # librosa dtw with normalized similarity
    D, wp = librosa.sequence.dtw(X=chromagram1, Y=chromagram2, subseq=subsequence, metric='euclidean')

    s = 0
    min_D = D.min()
    max_D = D.max()

    for i in range(len(wp)):
        x, y = wp[i]
        z = D[x, y]
        s += z

    avg_cost = s / len(wp)
    similarity = abs(1 - ((avg_cost - min_D) / (max_D - min_D)))

    return similarity


chroma_files = os.listdir(chroma_path)

# each element in final_maps contains [original file, generated midi/test file, closest match file]
final_maps = []

for og_file, gen_file in tqdm.tqdm(file_pairs):
    # load test file chromagram
    y, sr = librosa.load(gen_path + gen_file, sr=5000)
    reduced_noise_audio = nr.reduce_noise(y=y, sr=5000,time_mask_smooth_ms=52)
    sf.write('data/cleaned_audio/' + gen_file, reduced_noise_audio, 5000, subtype='PCM_24')
    chromagram1 = fetch_chromagram('data/cleaned_audio/' + gen_file, sr=5000, end_time=30)
    #chromagram1 = fetch_chromagram(gen_path + gen_file, sr=5000, end_time=30)
    scores = []
    for chroma_file in tqdm.tqdm(chroma_files):
        # fetch each train file chromagram
        chromagram2 = np.load(chroma_path + chroma_file)
        score1 = method1(chromagram1, chromagram2)
        score2 = method2(chromagram1, chromagram2)
        scores.append(score1)

    matched_file = chroma_files[np.argmin(scores)]  # argmin in case of cost, argmax in case of similarity

    # each element in final_maps contains [original file, generated midi/test file, closest match file]
    final_maps.append([og_file, gen_file, matched_file])

    break  # test for one test file print final_maps to check, remove break later

accuracy = 0
for i in final_maps:
    ground_truth = i[0][:-3]
    closest_match = i[2][:-3]
    accuracy += ground_truth == closest_match
accuracy = accuracy / len(final_maps)
print(accuracy * 100)
