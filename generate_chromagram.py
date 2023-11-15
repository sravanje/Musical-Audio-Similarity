"""
Generate Chromagrams of MusicNet custom tracks and save as png

- Dhawal Modi & Sravan Jayati
"""

import os
import matplotlib.pyplot as plt
import librosa.display
import argparse
from tqdm import tqdm

from chromagram import fetch_chromagram


def generate_chromagram(file_path, output_path):

    '''
    IMP: Need to change this to save chromagrams in numpy or pickle files instead of images
    '''

    # Fetch chromagram
    chromagram = fetch_chromagram(file_path)

    plt.figure(figsize=(15, 3))
    librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time')
    plt.title(f'Chromagram of {os.path.basename(file_path)}')
    plt.colorbar()
    plt.tight_layout()

    # Save the plot as a .png file if output_path is provided, otherwise display chromagram plot
    if output_path:
        plt.savefig(os.path.join(output_path, f'{os.path.basename(file_path)}_chromagram.png'))
    else:
        plt.show()

    plt.close()


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--folderorfile', 
                        default='folder', 
                        type = str,  
                        help ='"file" for generating chromagrams for single file; "folder" for generating chromagrams for all files within folder')
    parser.add_argument('-p', '--path', 
                        default='../musicnet/custom_generated_musicnet/', 
                        type = str,  
                        help ='path to file or folder')
    parser.add_argument('-op', '--outputpath', 
                        default='../Musical-Audio-Similarity/chromagrams/', 
                        type = str,  
                        help ='path to save chromagrams, use None to only display')

    args = parser.parse_args()
    path = args.path
    outputpath = args.outputpath if args.outputpath!='None' else None

    # Processing folder
    if args.folderorfile=='folder':
        for filename in tqdm(os.listdir(path)):
            if not filename.endswith('.wav'):
                continue
            file_path = os.path.join(path, filename)
            generate_chromagram(file_path, outputpath)
    
    # Processing file
    elif args.folderorfile=='file':
        generate_chromagram(path, outputpath)

    else:
        raise ValueError('"folderorfile" argument must be either "folder" or "file"')

