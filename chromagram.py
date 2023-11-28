"""
Function for chromagram, to be used in other files.

- Dhawal Modi & Sravan Jayati
"""

import librosa.display


def fetch_chromagram(file_path, sr=None, delay=0, end_time=None):
    """
    sr: chosen sampling rate
    delay: time in seconds, removes the first 'delay' seconds
    end_time: time in seconds, removes 'end_time' seconds at the end
    """

    # Load audio file
    y, sr = librosa.load(file_path, sr=sr)  # Must try lower sampling rates for faster dtw operation

    # Segment audio file
    if end_time is None:
        y = y[int(delay * sr):]
    else:
        y = y[int(delay * sr):int(delay * sr) + int(end_time * sr)]

    # Compute chromagram
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    # chromagram = librosa.feature.chroma_cens(y=y, sr=sr,hop_length = 1024)

    return chromagram
