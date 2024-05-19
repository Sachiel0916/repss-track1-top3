import csv
import os
import cv2
import matplotlib.pyplot as plt
import scipy.signal
import xlrd
import xlwt
import numpy as np
import math

window_size = 300


raw_dataset_name = 'vipl_v2'


def max_min_norm(x):
    if np.max(x) == np.min(x):
        y = (x - np.min(x)) * 255
    else:
        y = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
    return y


def node_map_norm(x):
    if len(x.shape) == 3:
        node_norm = np.zeros((x.shape[0], x.shape[1], x.shape[2]), dtype=np.uint8)
        for i in range(x.shape[0]):
            node_norm[i, :, 0] = max_min_norm(x[i, :, 0])
            node_norm[i, :, 1] = max_min_norm(x[i, :, 1])
            node_norm[i, :, 2] = max_min_norm(x[i, :, 2])
    elif len(x.shape) == 2:
        node_norm = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
        for i in range(x.shape[0]):
            node_norm[i, :] = max_min_norm(x[i, :])
    return node_norm


def generative_smooth_bvp(x, line_top):
    y = np.zeros((len(x)))
    for i in range(len(line_top) - 1):
        l = line_top[i + 1] - line_top[i]
        if l > 2:
            for k in range(l):
                w = l
                w = ((k / w) * 2) * math.pi
                w = math.cos(w)
                w = (w + 1) * 0.5
                y[line_top[i] + k] = w + 0.00001

    if line_top[0] != 0:
        if line_top[0] != 1:
            x_1 = x[0:line_top[0]]
            y[0:line_top[0]] = (x_1 - np.min(x_1)) / (np.max(x_1) - np.min(x_1))

    if line_top[len(line_top) - 1] != len(x):
        if line_top[len(line_top) - 1] != len(x) - 1:
            x_2 = x[line_top[len(line_top) - 1]: len(x)]
            y[line_top[len(line_top) - 1]: len(y)] = (x_2 - np.min(x_2)) / (np.max(x_2) - np.min(x_2))
    return y


def rgb2yuv(x):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2YUV)
    x = node_map_norm(x)
    return x


def gen_wave_map(x):
    x = max_min_norm(x)
    label_map = np.zeros((128, len(x)), dtype=np.uint8)
    for i in range(len(x)):
        label_map[:, i] = x[i]
    return label_map


def wave_map_norm(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    wave_top, _ = scipy.signal.find_peaks(x, prominence=(0.2,))
    if len(wave_top) != 0:
        ppg_with_top = np.zeros((len(x)))
        for i in range(len(wave_top)):
            if i == 0:
                ppg_with_top[0: wave_top[0]] = max_min_norm(x[0: wave_top[0]])
            if i > 0:
                ppg_with_top[wave_top[i - 1]: wave_top[i]] = max_min_norm(x[wave_top[i - 1]: wave_top[i]])
        if wave_top[len(wave_top) - 1] != len(ppg_with_top):
            ppg_with_top[wave_top[len(wave_top) - 1]:] = max_min_norm(x[wave_top[len(wave_top) - 1]:])
    elif len(wave_top) == 0:
        ppg_with_top = x
    return ppg_with_top


def compute_heart_rate(x, fps):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    wave_top, _ = scipy.signal.find_peaks(x, prominence=(0.2, 1))
    wave_spn = []
    for i in range(len(wave_top) - 1):
        x = wave_top[i + 1] - wave_top[i]
        wave_spn.append(x)
    mean_spn = np.mean(wave_spn)
    if len(wave_spn) == 0:
        mean_spn = 60 * fps / 80
    hr = int(60 / (mean_spn / fps))
    return hr


def rotate_90(x):
    x = cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE)
    return x


shift_num = 50
assert window_size % shift_num == 0
shift_drop_num = (window_size // shift_num) - 1


if raw_dataset_name == 'vipl_v2':
    save_i = 0
    save_dir = './node_map_dataset/' + raw_dataset_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir_sub_target_norm = save_dir + 'target_norm/'
    if not os.path.exists(save_dir_sub_target_norm):
        os.makedirs(save_dir_sub_target_norm)

    save_dir_sub_target_map = save_dir + 'target_map/'
    if not os.path.exists(save_dir_sub_target_map):
        os.makedirs(save_dir_sub_target_map)

    save_dir_sub_node_norm = save_dir + 'node_norm_yuv/'
    if not os.path.exists(save_dir_sub_node_norm):
        os.makedirs(save_dir_sub_node_norm)

    save_dir_sub_node_shuffle = save_dir + 'node_shuffle_yuv/'
    if not os.path.exists(save_dir_sub_node_shuffle):
        os.makedirs(save_dir_sub_node_shuffle)

    label_out = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet_out = label_out.add_sheet('sheet', cell_overwrite_ok=True)
    sheet_out.write(0, 0, 'target_hr')
    sheet_out.write(0, 1, 'compute_hr')
    sheet_out.write(0, 2, 'fps')
    sheet_out.write(1, 2, '30')

    dataset_dir = './vipl_v2/VV2_Main/'
    dataset_list = os.listdir(dataset_dir)
    dataset_list.sort()
    for sub_0 in dataset_list:
        sub_dir = dataset_dir + sub_0 + '/'
        sub_list = os.listdir(sub_dir)
        sub_list.sort()

        for video_0 in sub_list:
            the_dir = sub_dir + video_0
            if the_dir.endswith('.db'):
                os.remove(the_dir)

        sub_list = os.listdir(sub_dir)
        sub_list.sort()
        for video_0 in sub_list:
            node_map_path = sub_dir + video_0 + '/node_map.png'
            print(node_map_path)
            node_map = cv2.imread(node_map_path)

            target_ppg = []
            label_xls_path = sub_dir + video_0 + '/target_label.xls'
            label_xls = xlrd.open_workbook_xls(label_xls_path)
            sheet = 'sheet'
            sheet_label = label_xls.sheet_by_name(sheet)
            rows_label = sheet_label.nrows
            for i in range(rows_label):
                x = sheet_label.cell(i, 0).value
                target_ppg.append(x)
            target_ppg = target_ppg[1:]

            target_hr = float(sheet_label.cell(1, 1).value)
            vipl_v2_fps_one = float(sheet_label.cell(1, 3).value)
            vipl_v2_fps_one = 30

            assert len(target_ppg) == node_map.shape[1]

            node_norm = node_map_norm(node_map)
            np.random.shuffle(node_map)
            node_shuffle = node_map_norm(node_map)

            target_map = gen_wave_map(target_ppg)
            target_norm_map = gen_wave_map(wave_map_norm(target_ppg))

            k_mul = len(target_ppg) // shift_num
            if k_mul > 1:
                for k in range(k_mul - shift_drop_num):
                    compute_hr = compute_heart_rate(target_ppg[k * shift_num: k * shift_num + window_size], fps=vipl_v2_fps_one)
                    target_hr_seg = target_hr
                    sheet_out.write(save_i + 1, 0, target_hr_seg)
                    sheet_out.write(save_i + 1, 1, compute_hr)
                    save_i += 1

                    node_norm_seg = node_map_norm(node_norm[:, k * shift_num: k * shift_num + window_size])
                    node_norm_seg = rgb2yuv(node_norm_seg)
                    node_norm_seg = rotate_90(node_norm_seg)
                    cv2.imwrite(save_dir_sub_node_norm + sub_0 + '_' + video_0 + '_' + str(k).zfill(3)
                                + '_node_norm.png', node_norm_seg)

                    node_shuffle_seg = node_map_norm(node_shuffle[:, k * shift_num: k * shift_num + window_size])
                    node_shuffle_seg = rgb2yuv(node_shuffle_seg)
                    node_shuffle_seg = rotate_90(node_shuffle_seg)
                    cv2.imwrite(save_dir_sub_node_shuffle + sub_0 + '_' + video_0 + '_' + str(k).zfill(3)
                                + '_node_shuffle.png', node_shuffle_seg)

                    target_map_seg = node_map_norm(target_map[:, k * shift_num: k * shift_num + window_size])
                    target_map_seg = rotate_90(target_map_seg)
                    cv2.imwrite(save_dir_sub_target_map + sub_0 + '_' + video_0 + '_' + str(k).zfill(3)
                                + '_target_map.png', target_map_seg)

                    target_norm_map_seg = node_map_norm(target_norm_map[:, k * shift_num: k * shift_num + window_size])
                    target_norm_map_seg = rotate_90(target_norm_map_seg)
                    cv2.imwrite(save_dir_sub_target_norm + sub_0 + '_' + video_0 + '_' + str(k).zfill(3)
                                + '_target_norm.png', target_norm_map_seg)
    label_out.save(save_dir + 'hr_label.xls')
