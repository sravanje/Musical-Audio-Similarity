'''
Fetch and plot spectrogram

-Sravan Jayati
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile

path = '../tests/'

files = [path+'1727.wav',
         path+'1727_schubert_op114_2.wav',
         path+'1727_schubert_op114_2_whitenoise.wav',
         path+'1727_schubert_op114_2_crowd.wav']

Fs_list, aud_list = [], []
for file in files:
    Fs, aud = wavfile.read(file)
    if len(aud.shape)>1 and aud.shape[-1]>1:
        aud = aud[:,0]
    # audseg = aud[:int(Fs*125)]
    audseg = aud[:int(Fs*8)]
    aud_list.append(audseg)
    Fs_list.append(Fs)

fig = plt.figure()
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)
powerSpectrum, frequenciesFound, time, imageAxis = ax1.specgram(aud_list[0], Fs=Fs_list[0])
powerSpectrum, frequenciesFound, time, imageAxis = ax2.specgram(aud_list[1], Fs=Fs_list[1])
powerSpectrum, frequenciesFound, time, imageAxis = ax3.specgram(aud_list[2], Fs=Fs_list[2])
powerSpectrum, frequenciesFound, time, imageAxis = ax4.specgram(aud_list[3], Fs=Fs_list[3])
ax1.title.set_text('original file')
ax2.title.set_text('test file')
ax3.title.set_text('test file with whitenoise')
ax4.title.set_text('test file with crowd talking')
plt.show()