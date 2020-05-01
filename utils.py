from synthplayer import params
import itertools
import numpy as np
import time
import sounddevice as sd
import matplotlib.pyplot as plt


def get_raw_data(osc, duration=1, samplerate=44100):
    """Gets the raw data from synthplayer objects"""
    num_blocks = samplerate*duration//params.norm_osc_blocksize
    tmp = np.array(list(itertools.islice(osc.blocks(), num_blocks)))
    return tmp.flatten()


def play_sound(sound_array, sr=44100):
    """Plays the given sound"""
    duration = sound_array.shape[0] / sr
    sd.play(sound_array, sr)
    time.sleep(duration)


def plot_sound(sound_array):
    """Plots a few cycles of the given sound"""
    plt.figure(figsize=(10, 4))
    plt.plot(sound_array[:500])
    plt.show()
