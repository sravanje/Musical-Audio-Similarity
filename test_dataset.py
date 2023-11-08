'''
Converting midi files from musicnet dataset to wav files of different soundfonts.

-Sravan Jayati
'''

import os
from midi2mp3 import midi2wav
from tqdm import tqdm


def main(args):

    musicnet_path = args.musicnetpath
    musicnet_midi_path = musicnet_path+'musicnet_midis/'

    soundfont = args.sfpath
    save_path = musicnet_path+'custom_generated_musicnet/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    midi_filepaths = []
    for folder in os.listdir(musicnet_midi_path):
        subfolder_path = musicnet_midi_path+folder+"/"
        for file in os.listdir(subfolder_path):
            filepath = subfolder_path+file
            midi_filepaths.append(filepath)

    for midi_file in tqdm(midi_filepaths):
        save_filepath = save_path+midi_file.split('/')[-1][:-3]+'wav'
        if os.path.exists(save_filepath):
            continue
        midi2wav(midi_file, soundfont, save_filepath)


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--musicnetpath', 
                        default='../dataset3/archive/musicnet/', 
                        type = str,  
                        help ='path to midi file')
    parser.add_argument('-sf', '--sfpath', 
                        default="./soundfonts/Custom_Classical_Guitar.sf2", 
                        type = str, 
                        help ='path to soundfont file')

    args = parser.parse_args()

    main(args)
