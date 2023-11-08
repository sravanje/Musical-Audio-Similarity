'''
Given a midi file and a soundfont, converts midi file to wav file.

-Sravan Jayati
'''

import os
from midi2audio import FluidSynth


def midi2wav(midi_file, soundfont, out_path):
    '''
    midi_file: Path to midi file, extension '.mid', str
    soundfont: Path to soundfont file, extension '.sf2', str
    out_path:  Path to output file, extension '.wav', str
    '''
    if out_path.split('.')[-1]!='wav' or midi_file.split('.')[-1]!='mid' or soundfont.split('.')[-1]!='sf2':
        raise ValueError("midi must end in '.mid', soundfont must end in '.sf2', out_path must end in '.wav'")
    if not (os.path.exists(midi_file) and os.path.exists(soundfont)):
        raise ValueError("midi or soundfont files don't exist")
    
    fs = FluidSynth(soundfont)
    fs.midi_to_audio(midi_file, out_path)

    return


if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-md', '--midipath', 
                        default="./sample_files/Am_I_Blue_AB.mid", 
                        type = str,  
                        help ='path to midi file')
    parser.add_argument('-sf', '--sfpath', 
                        default="./soundfonts/Custom_Classical_Guitar.sf2", 
                        type = str, 
                        help ='path to soundfont file')
    parser.add_argument('-op', '--outpath', 
                        default="./sample_files/test.wav", 
                        type = str, 
                        help ='path to output wav file')

    args = parser.parse_args()

    print("\nmidipath: ", args.midipath, "\nsfpath: ", args.sfpath, "\noutpath: ", args.outpath, "\n")

    midi2wav(args.midipath, args.sfpath, args.outpath)
