# Feature extraction example
import numpy as np
import librosa
import librosa.display 
import matplotlib.pyplot as plt

# Load the example clip

y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=5)

librosa.feature.mfcc(y=y, sr=sr)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

librosa.feature.mfcc(S=librosa.power_to_db(S))

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()