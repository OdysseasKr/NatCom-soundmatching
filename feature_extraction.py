import numpy as np
import matplotlib.pyplot as plt
import librosa 
import librosa.feature
import librosa.display
from utils import plot_sound

sounds = np.load("dataset/sounds.npy")
sr = sounds.shape[1]
print(sounds.shape)

example_sound = sounds[0]
plot_sound(example_sound)

# MFCC
mfcc = librosa.feature.mfcc(example_sound, sr, n_mfcc=26//2)
print(mfcc.shape)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

# Centroid
centroid = librosa.feature.spectral_centroid(example_sound, sr)[0]
print(centroid.shape)

plt.figure(figsize=(10, 4))
plt.plot(centroid)
plt.title('Spectral centroid')
plt.tight_layout()
plt.show()

# Bandwidth
bandwidth = librosa.feature.spectral_bandwidth(example_sound, sr)[0]
print(bandwidth.shape)

plt.figure(figsize=(10, 4))
plt.plot(bandwidth)
plt.title('Spectral bandwidth')
plt.tight_layout()
plt.show()

# Flatness
flatness = librosa.feature.spectral_flatness(example_sound)[0]
print(flatness.shape)

plt.figure(figsize=(10, 4))
plt.plot(flatness)
plt.title('Spectral flatness')
plt.tight_layout()
plt.show()

# Roll-off
rolloff = librosa.feature.spectral_rolloff(example_sound)[0]
print(rolloff.shape)

plt.figure(figsize=(10, 4))
plt.plot(rolloff)
plt.title('Spectral rolloff')
plt.tight_layout()
plt.show()