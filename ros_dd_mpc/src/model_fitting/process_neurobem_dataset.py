import argparse
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.configuration_parameters import DirectoryConfig
from config.configuration_parameters import ModelFitConfig as Conf
from src.experiments.point_tracking_and_record import make_record_dict, jsonify
from src.quad_mpc.create_ros_dd_mpc import custom_quad_param_loader
from src.quad_mpc.quad_3d_mpc import Quad3DMPC
from src.utils.utils import safe_mkdir_recursive


def main(quad):
    full_path = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_processed_data')
    files = os.listdir(full_path)
    random.shuffle(files)

    rec_dicts = []
    for file in tqdm(files):
        try:
            rec_dict = make_record_dict(state_dim=13)
            process_file(os.path.join(full_path, file), quad, rec_dict)
            rec_dicts.append(rec_dict)
        except Exception as e:
            print(e)

    rec_dict = {}
    rec_dict['state_in'] = np.concatenate([data_dict['state_in'] for data_dict in rec_dicts], axis=0)
    rec_dict['input_in'] = np.concatenate([data_dict['input_in'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_out'] = np.concatenate([data_dict['state_out'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_ref'] = np.concatenate([data_dict['state_ref'] for data_dict in rec_dicts], axis=0)
    rec_dict['timestamp'] = np.concatenate([data_dict['timestamp'] for data_dict in rec_dicts], axis=0)
    rec_dict['dt'] = np.concatenate([data_dict['dt'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_pred'] = np.concatenate([data_dict['state_pred'] for data_dict in rec_dicts], axis=0)
    rec_dict['error'] = np.concatenate([data_dict['error'] for data_dict in rec_dicts], axis=0)

    del rec_dicts

    # Save datasets
    save_file_folder = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_dataset', 'train')
    safe_mkdir_recursive(save_file_folder)
    save_file = os.path.join(save_file_folder, f'npz_dataset_001.npz')
    np.savez(save_file, **rec_dict)


    # # Validation
    # rec_dicts = []
    # for file in tqdm(files[-20:]):
    #     try:
    #         rec_dict = make_record_dict(state_dim=13)
    #         process_file(os.path.join(full_path, file), quad, rec_dict)
    #         rec_dicts.append(rec_dict)
    #     except Exception as e:
    #         print(e)
    #
    # rec_dict = {}
    # rec_dict['state_in'] = np.concatenate([data_dict['state_in'] for data_dict in rec_dicts], axis=0)
    # rec_dict['input_in'] = np.concatenate([data_dict['input_in'] for data_dict in rec_dicts], axis=0)
    # rec_dict['state_out'] = np.concatenate([data_dict['state_out'] for data_dict in rec_dicts], axis=0)
    # rec_dict['state_ref'] = np.concatenate([data_dict['state_ref'] for data_dict in rec_dicts], axis=0)
    # rec_dict['timestamp'] = np.concatenate([data_dict['timestamp'] for data_dict in rec_dicts], axis=0)
    # rec_dict['dt'] = np.concatenate([data_dict['dt'] for data_dict in rec_dicts], axis=0)
    # rec_dict['state_pred'] = np.concatenate([data_dict['state_pred'] for data_dict in rec_dicts], axis=0)
    # rec_dict['error'] = np.concatenate([data_dict['error'] for data_dict in rec_dicts], axis=0)
    #
    # del rec_dicts
    #
    # # Save datasets
    # save_file_folder = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_dataset', 'test')
    # safe_mkdir_recursive(save_file_folder)
    # save_file = os.path.join(save_file_folder, f'npz_dataset_001.npz')
    # np.savez(save_file, **rec_dict)

def val(quad):
    full_path = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_processed_data')
    val_files = ['merged_2021-02-23-17-35-26_seg_1.csv', 'merged_2021-02-23-22-54-17_seg_1.csv']

    rec_dicts = []
    for file in tqdm(val_files):
        try:
            rec_dict = make_record_dict(state_dim=13)
            process_file(os.path.join(full_path, file), quad, rec_dict)
            rec_dicts.append(rec_dict)
        except Exception as e:
            print(e)

    rec_dict = {}
    rec_dict['state_in'] = np.concatenate([data_dict['state_in'] for data_dict in rec_dicts], axis=0)
    rec_dict['input_in'] = np.concatenate([data_dict['input_in'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_out'] = np.concatenate([data_dict['state_out'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_ref'] = np.concatenate([data_dict['state_ref'] for data_dict in rec_dicts], axis=0)
    rec_dict['timestamp'] = np.concatenate([data_dict['timestamp'] for data_dict in rec_dicts], axis=0)
    rec_dict['dt'] = np.concatenate([data_dict['dt'] for data_dict in rec_dicts], axis=0)
    rec_dict['state_pred'] = np.concatenate([data_dict['state_pred'] for data_dict in rec_dicts], axis=0)
    rec_dict['error'] = np.concatenate([data_dict['error'] for data_dict in rec_dicts], axis=0)

    del rec_dicts

    # Save datasets
    save_file_folder = os.path.join(DirectoryConfig.DATA_DIR, 'neurobem_dataset', 'test')
    safe_mkdir_recursive(save_file_folder)
    save_file = os.path.join(save_file_folder, f'npz_dataset_001.npz')
    np.savez(save_file, **rec_dict)





def process_file(file_path, quad, rec_dict):

    data = pd.read_csv(file_path, encoding='latin-1')
    for x_0, x_f, u, dt in consecutive_data_points(data):
        resimulate(x_0, x_f, u, dt, quad, rec_dict)


def consecutive_data_points(data):
    for i in range(0, len(data)-1, 1):
        data_0 = data.iloc[i]
        data_1 = data.iloc[i+1]
        t0 = data_0['t']
        t1 = data_1['t']
        dt = t1 - t0

        x_0 = np.hstack([
            data_0['pos x'],
            data_0['pos y'],
            data_0['pos z'],
            data_0['quat w'],
            data_0['quat x'],
            data_0['quat y'],
            data_0['quat z'],
            data_0['vel x'],
            data_0['vel y'],
            data_0['vel z'],
            data_0['ang vel x'],
            data_0['ang vel y'],
            data_0['ang vel z'],
        ])

        x_1 = np.hstack([
            data_1['pos x'],
            data_1['pos y'],
            data_1['pos z'],
            data_1['quat w'],
            data_1['quat x'],
            data_1['quat y'],
            data_1['quat z'],
            data_1['vel x'],
            data_1['vel y'],
            data_1['vel z'],
            data_1['ang vel x'],
            data_1['ang vel y'],
            data_1['ang vel z'],
        ])

        u = np.hstack([
            data_0['mot 2'],
            data_0['mot 3'],
            data_0['mot 1'],
            data_0['mot 4'],
        ])

        u = u ** 2 * quad.thrust_map[0] / quad.max_thrust

        yield x_0, x_1, u, dt


def resimulate(x_0, x_f, u, dt, quad_mpc, rec_dict):
    x_pred, _ = quad_mpc.forward_prop(x_0, u, t_horizon=dt, use_gp=False)
    x_pred = x_pred[-1, np.newaxis, :]

    rec_dict['state_in'] = np.append(rec_dict['state_in'], x_0[np.newaxis, :], axis=0)
    rec_dict['input_in'] = np.append(rec_dict['input_in'], u[np.newaxis, :], axis=0)
    rec_dict['state_out'] = np.append(rec_dict['state_out'], x_f[np.newaxis, :], axis=0)
    rec_dict['state_ref'] = np.append(rec_dict['state_ref'], np.zeros_like(x_f[np.newaxis, :]), axis=0)
    rec_dict['timestamp'] = np.append(rec_dict['timestamp'], np.zeros_like(dt))
    rec_dict['dt'] = np.append(rec_dict['dt'], dt)
    rec_dict['state_pred'] = np.append(rec_dict['state_pred'], x_pred, axis=0)
    rec_dict['error'] = np.append(rec_dict['error'], x_f - x_pred, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--quad", type=str, default="kingfisher",
                        help="Name of the quad.")

    input_arguments = parser.parse_args()

    ds_name = Conf.ds_name
    simulation_options = Conf.ds_metadata

    quad = custom_quad_param_loader(input_arguments.quad)
    quad_mpc = Quad3DMPC(quad)

    main(quad_mpc)
    val(quad_mpc)
