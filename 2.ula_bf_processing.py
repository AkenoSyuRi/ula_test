from pathlib import Path

import numpy as np
from scipy import signal
from tqdm import tqdm

from comm import get_mic_coords
from lzf_utils.audio_utils import AudioReader, AudioWriter
from lzf_utils.stft_utils import Stft


class Beamformer:
    def __init__(self, mic_coords, win_size=512, hop_size=256, sr=16000, c=340, delta_angle=5, sll=45):
        self.mic_coords = mic_coords
        self.win_size = win_size
        self.hop_size = hop_size
        self.fft_bins = win_size // 2 + 1
        self.sr = sr
        self.c = c
        self.N = mic_coords.shape[0]
        self.f_max = sr / 2
        self.f_res = sr / win_size

        self.deg_az_range = np.arange(0, 181, delta_angle)
        self.A = self.steering_vector()

        self.taper = signal.windows.chebwin(self.N, sll)[..., None]
        self.taper = self.taper / np.sum(self.taper) * self.N

    def steering_vector(self):
        rad_az_range = np.deg2rad(self.deg_az_range)[None]
        f_range = np.arange(0, self.f_max + 1, self.f_res).reshape(-1, 1, 1)
        d = self.mic_coords[:, 0, None] * np.cos(rad_az_range) + self.mic_coords[:, 1, None] * np.sin(rad_az_range)

        A = np.exp(1j * 2 * np.pi * f_range * d / self.c)  # (f, N, θ)
        return A

    def process(self, in_spec, doa):
        """(N, F), θ"""
        ind_az_tar = np.argwhere(self.deg_az_range == doa)[0][0]
        A_tar_H = self.A[..., ind_az_tar].T.conj()

        out_spec0 = np.mean(in_spec, 0)
        out_spec1 = np.mean(in_spec * A_tar_H, 0)
        out_spec2 = np.mean(in_spec * A_tar_H * self.taper, 0)

        out_spec = np.row_stack([out_spec0, out_spec1, out_spec2])
        return out_spec


def main():
    in_wav_dir = Path(r"D:\Temp\pra_sim_out\sim_dynamic_anechoic_snr20")
    # in_wav_dir = Path(r"D:\Temp\pra_sim_out\sim_dynamic_rt60_0.2s_snr20")
    out_wav_dir = Path(r"D:\Temp\athena_test_out")
    in_channels, out_channels, az_tar = 11, 3, 90
    out_name1 = f"{in_wav_dir.name}_{az_tar=}_inp"
    out_name2 = f"{in_wav_dir.name}_{az_tar=}_out"

    beamformer = Beamformer(get_mic_coords()[0], )  # use default config
    ar = AudioReader(in_wav_dir, beamformer.sr)
    aw = AudioWriter(out_wav_dir, beamformer.sr)
    stft = Stft(beamformer.win_size, beamformer.hop_size, in_channels, out_channels)

    for i, in_data in tqdm(enumerate(ar.read_audio_data(beamformer.hop_size))):
        in_spec = stft.transform(in_data)
        out_spec = beamformer.process(in_spec, az_tar)
        out_data = stft.inverse(out_spec)

        aw.write_data_list(out_name1, [in_data[0]])
        if i > 0:
            aw.write_data_list(out_name2, out_data)
    aw.write_data_list(out_name2, np.zeros([out_channels, beamformer.hop_size]))
    ...


if __name__ == '__main__':
    main()
    ...
