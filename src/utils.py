from synthplayer import params
import itertools
import numpy as np
import time
import sounddevice as sd
import matplotlib.pyplot as plt
import json


COLOURS = ["#e6194B","#3cb44b", "#ffe119", "#4363d8", "#f58231", "#42d4f4", "#f032e6", "#fabebe", "#469990", "#e6beff", "#9A6324", "#800000", "#aaffc3", "#000075", "#a9a9a9"]


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


def int_to_bin(x):
    """
        Converts integer to list of binary digits
    """
    return list(str(bin(x))[2:])


def bin_to_int(x):
    """
        Converts a list of binary digits to an integer
    """
    x = ''.join(x)
    return int(x, 2)


def num_of_digits(x):
    """
        Calculates how many digits are necessary to represent x in binary
    """
    return len(int_to_bin(x))


def plot_results(path):
    data = json.load(open(path))
    for t,target in enumerate(data["targets"]):
        figure = plt.figure()
        plt.grid(zorder=1)
        for i,run in enumerate(target["runs"]):
            best_fits, worst_fits = [], []
            for generation in run["gen_stats"]:
                best_fits.append(generation["best"])
                worst_fits.append(generation["worst"])
            c = COLOURS[i]
            plt.plot(best_fits, label=f"Run {i}", zorder=2, color=c, linestyle="solid", linewidth=3)
            plt.plot(worst_fits, zorder=2, color=c, linestyle="dashed", linewidth=3)
            if run["early_stopping"]:
                plt.plot(len(best_fits)-1, 5, 'bo', color=c, markersize=10, zorder=3)
        plt.ylabel("Fitness (MSE)")
        plt.ylim((0,500))
        plt.xlabel("Generations")
        plt.title(f"Target {t}")
        plt.legend()
        plt.show()