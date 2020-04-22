from synthplayer import params
import itertools
import numpy as np

def get_data(osc, duration=1, samplerate=44100):
    num_blocks = samplerate*duration//params.norm_osc_blocksize
    tmp = np.array(list(itertools.islice(osc.blocks(), num_blocks)))
    return tmp.flatten()
