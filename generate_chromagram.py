"""
Generate Chromagrams of MusicNet custom tracks and save as png
- Dhawal Modi & Sravan Jayati
"""

import os
import matplotlib.pyplot as plt
import librosa.display

folder_path = 'C:/Users/Dave/Documents/EECS257/Project/Signals_project/dataset3/archive/musicnet/custom_generated_musicnet/'
output_path = 'C:/Users/Dave/Documents/EECS257/Project/Signals_project/Musical-Audio-Similarity/chromagrams'

for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):
        file_path = os.path.join(folder_path, filename)

    # Load two audio files
    y1, sr1 = librosa.load(file_path,sr = None)


    # Compute chromagrams
    chromagram = librosa.feature.chroma_stft(y=y1, sr=sr1)


    # Display chromagrams
    plt.figure(figsize=(15, 5))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
    plt.title(f'Chromagram of {filename}')
    plt.colorbar()

    plt.tight_layout()
    
    # Save the plot as a .png file
    plt.savefig(os.path.join(output_path, f'{filename}_chromagram.png'))
    plt.close()
