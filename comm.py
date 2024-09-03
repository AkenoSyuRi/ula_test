import numpy as np


def get_mic_coords(N=11, D=0.1):
    mic_coords = np.zeros([N, 3])
    mic_ids = np.arange(N)

    x = mic_ids * D
    mic_coords[:, 0] = x - np.mean(x)

    return mic_coords, mic_ids