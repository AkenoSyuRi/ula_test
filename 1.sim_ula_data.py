from pathlib import Path

import numpy as np

from comm import get_mic_coords
from lzf_utils.sim_utils import RoomDataSimulator, get_audio_signal


def get_test_src_pos_list(mic0_coord, deg_angles=np.arange(0, 91, 20)):
    rad_angles = np.deg2rad(deg_angles)

    r = min(mic0_coord[0], mic0_coord[1]) - 0.1
    x = mic0_coord[0] + r * np.cos(rad_angles)
    y = mic0_coord[1] + r * np.sin(rad_angles)
    z = np.tile(mic0_coord[2], rad_angles.shape[0])

    src_pos_list = np.column_stack([x, y, z])
    return src_pos_list


def sim_fixed_src_pos(ref_rt60=None, ref_snr=None):
    save_pcm, save_mono, out_name = bool(0), bool(0), "sim"
    fs, room_size, out_db = 16000, [8, 5, 3.5], -50

    mic0_coord = [room_size[0] / 2, room_size[1] / 2, 1.0]
    mic_coords = get_mic_coords()[0] + np.array(mic0_coord)  # mic0: 4,2.5,3.4
    # PlotUtils.plot_3d_coord(mic_coords)  # TODO: debug

    src_pos_list = [
        [mic0_coord[0], 0.5, 1.3],
        [0.5, 0.5, 1.3],
    ]
    src_path_list = [
        r"F:\Test\4.play\anechoic_chamber\007537_16k.wav",
        r"F:\Test\4.play\anechoic_chamber\lzf_speech_cut9s_16k.wav",
    ]
    out_dir = Path(r"D:\Temp\pra_sim_out")
    # =======================================================================

    rds = RoomDataSimulator(room_size, mic_coords.T, fs, snr=ref_snr, rt60=ref_rt60)
    sig_infos = rds.map2sig_infos(src_path_list, src_pos_list)
    rds.simulate(*sig_infos)
    rds.save(out_dir, out_name, out_db=out_db, mono=save_mono, save_pcm=save_pcm)
    ...


def sim_dynamic_src_pos(ref_rt60=None, ref_snr=None):
    save_pcm, save_mono, out_name = bool(0), bool(1), "sim_dynamic"
    fs, room_size, out_db = 16000, [7, 4, 3], -23
    dynamic_deg_angles = np.arange(0, 181, 10)

    mic0_coord = [room_size[0] / 2, room_size[1] / 2, 1.0]
    mic_coords = get_mic_coords()[0] + np.array(mic0_coord)  # mic0: 4,2.5,3.4

    dynamic_src_pos_list = get_test_src_pos_list(mic0_coord, dynamic_deg_angles)
    dynamic_src_path = r"F:\Test\4.play\anechoic_chamber\007537_16k.wav"
    out_dir = Path(r"D:\Temp\pra_sim_out")
    # =======================================================================

    in_wav_data = get_audio_signal(dynamic_src_path, fs)
    aw, delay, delay2 = None, 0, 0.5
    for dynamic_src_pos in dynamic_src_pos_list:
        rds = RoomDataSimulator(room_size, mic_coords.T, fs, snr=ref_snr, rt60=ref_rt60)
        sig_infos = rds.map2sig_infos([in_wav_data], [dynamic_src_pos], delay=delay)
        rds.simulate(*sig_infos)
        aw = rds.save(
            out_dir,
            out_name,
            out_db=out_db,
            mono=save_mono,
            save_pcm=save_pcm,
            audio_writer=aw,
        )
        delay = delay2
        del rds
    ...


def main():
    # sim_fixed_src_pos(ref_rt60=0.3, ref_snr=20)

    sim_dynamic_src_pos(None, 20)
    sim_dynamic_src_pos(0.2, 20)
    ...


if __name__ == "__main__":
    main()
    ...
