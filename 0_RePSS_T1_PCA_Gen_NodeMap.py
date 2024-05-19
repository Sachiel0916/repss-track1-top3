import csv
import cv2
import os
import numpy as np
import pandas as pd
import xlrd
import face_recognition


learn_root = './vipl_v2/VV2_Main/'
learn_list = os.listdir(learn_root)
learn_list.sort()

learn_dataset = []

data_total_xls = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet_total = data_total_xls.add_sheet('sheet', cell_overwrite_ok=True)

k_ = 0
for i in range(len(learn_list)):
    sub_dir = learn_root + learn_list[i] + '/'
    sub_list = os.listdir(sub_dir)
    sub_list.sort()

    for j in range(len(sub_list)):
        video_dir = sub_dir + sub_list[j] + '/'
        learn_dataset.append(video_dir)
        sheet_total.write(k_, 0, video_dir)
        k_ += 1

data_total_xls.save('./vipl_v/VV2_data.xls')


for j in range(len(learn_dataset)):
    frame_dir = learn_dataset[j] + 'face/'
    frame_list = os.listdir(frame_dir)
    frame_list.sort()

    label_xls = xlrd.open_workbook_xls(learn_dataset[j] + 'target_label.xls')
    sheet = 'sheet'
    sheet_label = label_xls.sheet_by_name(sheet)
    frame_count = int(sheet_label.cell(1, 2).value)
    rows_label = sheet_label.nrows
    assert rows_label == 301

    assert len(frame_list) >= frame_count

    node_map = np.zeros((128, frame_count, 3))

    for k in range(len(frame_list)):

        img_path = frame_dir + frame_list[k]
        if img_path.endswith('.jpg'):
            print(img_path)
            img_raw = cv2.imread(img_path)
            raw_h = img_raw.shape[1]
            if raw_h == 1280:
                dis_parameter = 200
            else:
                dis_parameter = 100

            img_pil = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

            if k == 0:
                face_location = face_recognition.face_locations(img_pil, model='hog')
                if len(face_location) == 0:
                    face_location = face_recognition.face_locations(img_pil, model='cnn')

                    if len(face_location) == 0:
                        img_last_one = cv2.imread(learn_dataset[j - 1] + 'face/Frame_0001.jpg')
                        img_last_one = cv2.cvtColor(img_last_one, cv2.COLOR_BGR2RGB)
                        face_location = face_recognition.face_locations(img_last_one, model='cnn')

                    x_spn = face_location[0][1] - face_location[0][3]
                    y_spn = face_location[0][2] - face_location[0][0]
                    min_spn = np.min([x_spn, y_spn])
                    if min_spn < dis_parameter:
                        img_pil[face_location[0][0]: face_location[0][2], face_location[0][3]: face_location[0][1]] = 0
                        face_location = face_recognition.face_locations(img_pil, model='cnn')

                elif len(face_location) != 0:
                    x_spn = face_location[0][1] - face_location[0][3]
                    y_spn = face_location[0][2] - face_location[0][0]
                    min_spn = np.min([x_spn, y_spn])
                    if min_spn < dis_parameter:
                        face_location = face_recognition.face_locations(img_pil, model='cnn')

                face_landmarks_list = face_recognition.face_landmarks(img_pil, face_locations=face_location)
                face_have_landmarks_list = face_landmarks_list

            else:
                face_location = face_recognition.face_locations(img_pil, model='hog')
                if len(face_location) != 0:
                    x_spn = face_location[0][1] - face_location[0][3]
                    y_spn = face_location[0][2] - face_location[0][0]
                    min_spn = np.min([x_spn, y_spn])
                    if min_spn > dis_parameter:
                        face_landmarks_list = face_recognition.face_landmarks(img_pil)
                        face_have_landmarks_list = face_landmarks_list
                    else:
                        face_landmarks_list = face_have_landmarks_list
                else:
                    face_landmarks_list = face_have_landmarks_list

            for local_land in face_landmarks_list:
                landmarks_raw = local_land['chin'] + local_land['left_eyebrow'] + local_land['right_eyebrow'] + \
                                local_land['left_eye'] + local_land['right_eye'] + local_land['nose_bridge'] + \
                                local_land['nose_tip'] + local_land['top_lip'] + local_land['bottom_lip']

            landmarks = landmarks_raw[0:60] + landmarks_raw[61:66] + landmarks_raw[68:71]

            l_x = []
            l_y = []
            for l in range(len(landmarks)):
                l_x.append(landmarks[l][0])
                l_y.append(landmarks[l][1])

            if np.max(l_x) > img_raw.shape[1]:
                x_max = img_raw.shape[1]
            else:
                x_max = np.max(l_x)

            if np.min(l_x) < 0:
                x_min = 0
            else:
                x_min = np.min(l_x)

            if np.max(l_y) > img_raw.shape[0]:
                y_max = img_raw.shape[0]
            else:
                y_max = np.max(l_y)

            if np.min(l_y) < 0:
                y_min = 0
            else:
                y_min = np.min(l_y)

            x_spn = x_max - x_min
            y_spn = y_max - y_min

            if x_spn < 20:
                x_spn = 20
            if y_spn < 20:
                y_spn = 20

            x_0 = (landmarks[0][0] + landmarks[27][0]) // 2
            y_0 = (landmarks[0][1] + landmarks[27][1]) // 2

            x_1 = (landmarks[0][0] + landmarks[43][0]) // 2
            y_1 = (landmarks[0][1] + landmarks[43][1]) // 2

            x_2 = (landmarks[0][0] + landmarks[48][0]) // 2
            y_2 = (landmarks[0][1] + landmarks[48][1]) // 2

            x_3 = (landmarks[1][0] + landmarks[40][0]) // 2
            y_3 = (landmarks[1][1] + landmarks[40][1]) // 2

            x_4 = (landmarks[1][0] + landmarks[41][0]) // 2
            y_4 = (landmarks[1][1] + landmarks[41][1]) // 2

            x_5 = (landmarks[1][0] + landmarks[43][0]) // 2
            y_5 = (landmarks[1][1] + landmarks[43][1]) // 2

            x_6 = (landmarks[2][0] + landmarks[40][0]) // 2
            y_6 = (landmarks[2][1] + landmarks[40][1]) // 2

            x_7 = (landmarks[2][0] + landmarks[41][0]) // 2
            y_7 = (landmarks[2][1] + landmarks[41][1]) // 2

            x_8 = (landmarks[2][0] + landmarks[43][0]) // 2
            y_8 = (landmarks[2][1] + landmarks[43][1]) // 2

            x_9 = (landmarks[2][0] + landmarks[48][0]) // 2
            y_9 = (landmarks[2][1] + landmarks[48][1]) // 2

            x_10 = (landmarks[3][0] + landmarks[40][0]) // 2
            y_10 = (landmarks[3][1] + landmarks[40][1]) // 2

            x_11 = (landmarks[3][0] + landmarks[41][0]) // 2
            y_11 = (landmarks[3][1] + landmarks[41][1]) // 2

            x_12 = (landmarks[3][0] + landmarks[43][0]) // 2
            y_12 = (landmarks[3][1] + landmarks[43][1]) // 2

            x_13 = (landmarks[3][0] + landmarks[48][0]) // 2
            y_13 = (landmarks[3][1] + landmarks[48][1]) // 2

            x_14 = (landmarks[4][0] + landmarks[27][0]) // 2
            y_14 = (landmarks[4][1] + landmarks[27][1]) // 2

            x_15 = (landmarks[4][0] + landmarks[32][0]) // 2
            y_15 = (landmarks[4][1] + landmarks[32][1]) // 2

            x_16 = (landmarks[4][0] + landmarks[31][0]) // 2
            y_16 = (landmarks[4][1] + landmarks[31][1]) // 2

            x_17 = (landmarks[4][0] + landmarks[30][0]) // 2
            y_17 = (landmarks[4][1] + landmarks[30][1]) // 2

            x_18 = (landmarks[4][0] + landmarks[43][0]) // 2
            y_18 = (landmarks[4][1] + landmarks[43][1]) // 2

            x_19 = (landmarks[4][0] + landmarks[48][0]) // 2
            y_19 = (landmarks[4][1] + landmarks[48][1]) // 2

            x_20 = (landmarks[5][0] + landmarks[32][0]) // 2
            y_20 = (landmarks[5][1] + landmarks[32][1]) // 2

            x_21 = (landmarks[5][0] + landmarks[31][0]) // 2
            y_21 = (landmarks[5][1] + landmarks[31][1]) // 2

            x_22 = (landmarks[5][0] + landmarks[48][0]) // 2
            y_22 = (landmarks[5][1] + landmarks[48][1]) // 2

            x_23 = (landmarks[6][0] + landmarks[31][0]) // 2
            y_23 = (landmarks[6][1] + landmarks[31][1]) // 2

            x_24 = (landmarks[6][0] + landmarks[48][0]) // 2
            y_24 = (landmarks[6][1] + landmarks[48][1]) // 2

            x_25 = (landmarks[6][0] + landmarks[64][0]) // 2
            y_25 = (landmarks[6][1] + landmarks[64][1]) // 2

            x_26 = (landmarks[7][0] + landmarks[63][0]) // 2
            y_26 = (landmarks[7][1] + landmarks[63][1]) // 2

            x_27 = (landmarks[8][0] + landmarks[62][0]) // 2
            y_27 = (landmarks[8][1] + landmarks[62][1]) // 2

            x_28 = (landmarks[9][0] + landmarks[61][0]) // 2
            y_28 = (landmarks[9][1] + landmarks[61][1]) // 2

            x_29 = (landmarks[10][0] + landmarks[60][0]) // 2
            y_29 = (landmarks[10][1] + landmarks[60][1]) // 2

            x_30 = (landmarks[10][0] + landmarks[54][0]) // 2
            y_30 = (landmarks[10][1] + landmarks[54][1]) // 2

            x_31 = (landmarks[10][0] + landmarks[38][0]) // 2
            y_31 = (landmarks[10][1] + landmarks[38][1]) // 2

            x_32 = (landmarks[11][0] + landmarks[54][0]) // 2
            y_32 = (landmarks[11][1] + landmarks[54][1]) // 2

            x_33 = (landmarks[11][0] + landmarks[38][0]) // 2
            y_33 = (landmarks[11][1] + landmarks[38][1]) // 2

            x_34 = (landmarks[11][0] + landmarks[37][0]) // 2
            y_34 = (landmarks[11][1] + landmarks[37][1]) // 2

            x_35 = (landmarks[12][0] + landmarks[54][0]) // 2
            y_35 = (landmarks[12][1] + landmarks[54][1]) // 2

            x_36 = (landmarks[12][0] + landmarks[47][0]) // 2
            y_36 = (landmarks[12][1] + landmarks[47][1]) // 2

            x_37 = (landmarks[12][0] + landmarks[33][0]) // 2
            y_37 = (landmarks[12][1] + landmarks[33][1]) // 2

            x_38 = (landmarks[12][0] + landmarks[38][0]) // 2
            y_38 = (landmarks[12][1] + landmarks[38][1]) // 2

            x_39 = (landmarks[12][0] + landmarks[37][0]) // 2
            y_39 = (landmarks[12][1] + landmarks[37][1]) // 2

            x_40 = (landmarks[12][0] + landmarks[36][0]) // 2
            y_40 = (landmarks[12][1] + landmarks[36][1]) // 2

            x_41 = (landmarks[13][0] + landmarks[54][0]) // 2
            y_41 = (landmarks[13][1] + landmarks[54][1]) // 2

            x_42 = (landmarks[13][0] + landmarks[47][0]) // 2
            y_42 = (landmarks[13][1] + landmarks[47][1]) // 2

            x_43 = (landmarks[13][0] + landmarks[41][0]) // 2
            y_43 = (landmarks[13][1] + landmarks[41][1]) // 2

            x_44 = (landmarks[13][0] + landmarks[40][0]) // 2
            y_44 = (landmarks[13][1] + landmarks[40][1]) // 2

            x_45 = (landmarks[14][0] + landmarks[54][0]) // 2
            y_45 = (landmarks[14][1] + landmarks[54][1]) // 2

            x_46 = (landmarks[14][0] + landmarks[47][0]) // 2
            y_46 = (landmarks[14][1] + landmarks[47][1]) // 2

            x_47 = (landmarks[14][0] + landmarks[41][0]) // 2
            y_47 = (landmarks[14][1] + landmarks[41][1]) // 2

            x_48 = (landmarks[14][0] + landmarks[40][0]) // 2
            y_48 = (landmarks[14][1] + landmarks[40][1]) // 2

            x_49 = (landmarks[15][0] + landmarks[47][0]) // 2
            y_49 = (landmarks[15][1] + landmarks[47][1]) // 2

            x_50 = (landmarks[15][0] + landmarks[41][0]) // 2
            y_50 = (landmarks[15][1] + landmarks[41][1]) // 2

            x_51 = (landmarks[15][0] + landmarks[40][0]) // 2
            y_51 = (landmarks[15][1] + landmarks[40][1]) // 2

            x_52 = (landmarks[16][0] + landmarks[54][0]) // 2
            y_52 = (landmarks[16][1] + landmarks[54][1]) // 2

            x_53 = (landmarks[16][0] + landmarks[47][0]) // 2
            y_53 = (landmarks[16][1] + landmarks[47][1]) // 2

            x_54 = (landmarks[16][0] + landmarks[36][0]) // 2
            y_54 = (landmarks[16][1] + landmarks[36][1]) // 2

            x_55 = (landmarks[21][0] + landmarks[22][0]) // 2
            y_55 = (landmarks[21][1] + landmarks[22][1]) // 2

            x_56 = (landmarks[20][0] + landmarks[23][0]) // 2
            y_56 = (landmarks[20][1] + landmarks[23][1]) // 2

            x_57 = (landmarks[19][0] + landmarks[24][0]) // 2
            y_57 = (landmarks[19][1] + landmarks[24][1]) // 2

            x_58 = (landmarks[42][0] + landmarks[41][0]) // 2
            y_58 = (landmarks[42][1] + landmarks[41][1]) // 2

            x_59 = (landmarks[43][0] + landmarks[49][0]) // 2
            y_59 = (landmarks[43][1] + landmarks[49][1]) // 2

            x_60 = (landmarks[44][0] + landmarks[50][0]) // 2
            y_60 = (landmarks[44][1] + landmarks[50][1]) // 2

            x_61 = (landmarks[45][0] + landmarks[51][0]) // 2
            y_61 = (landmarks[45][1] + landmarks[51][1]) // 2

            x_62 = (landmarks[46][0] + landmarks[52][0]) // 2
            y_62 = (landmarks[46][1] + landmarks[52][1]) // 2

            x_63 = (landmarks[47][0] + landmarks[53][0]) // 2
            y_63 = (landmarks[47][1] + landmarks[53][1]) // 2

            x_64 = (landmarks[4][0] + landmarks[17][0]) // 2
            y_64 = (landmarks[4][1] + landmarks[17][1]) // 2

            x_65 = (landmarks[12][0] + landmarks[26][0]) // 2
            y_65 = (landmarks[12][1] + landmarks[26][1]) // 2

            x_66 = (landmarks[40][0] + landmarks[41][0]) // 2
            y_66 = (landmarks[40][1] + landmarks[41][1]) // 2

            x_67 = (landmarks[14][0] + landmarks[26][0]) // 2
            y_67 = (landmarks[14][1] + landmarks[26][1]) // 2

            x_68 = (landmarks[2][0] + landmarks[17][0]) // 2
            y_68 = (landmarks[2][1] + landmarks[17][1]) // 2

            x_69 = (((landmarks[1][0] + landmarks[40][0]) // 2) + landmarks[1][0]) // 2
            y_69 = (((landmarks[1][1] + landmarks[40][1]) // 2) + landmarks[1][1]) // 2

            x_70 = (((landmarks[1][0] + landmarks[40][0]) // 2) + landmarks[40][0]) // 2
            y_70 = (((landmarks[1][1] + landmarks[40][1]) // 2) + landmarks[40][1]) // 2

            x_71 = (((landmarks[15][0] + landmarks[40][0]) // 2) + landmarks[15][0]) // 2
            y_71 = (((landmarks[15][1] + landmarks[40][1]) // 2) + landmarks[15][1]) // 2

            x_72 = (((landmarks[15][0] + landmarks[40][0]) // 2) + landmarks[40][0]) // 2
            y_72 = (((landmarks[15][1] + landmarks[40][1]) // 2) + landmarks[40][1]) // 2

            x_73 = (((landmarks[2][0] + landmarks[41][0]) // 2) + landmarks[2][0]) // 2
            y_73 = (((landmarks[2][1] + landmarks[41][1]) // 2) + landmarks[2][1]) // 2

            x_74 = (((landmarks[2][0] + landmarks[41][0]) // 2) + landmarks[41][0]) // 2
            y_74 = (((landmarks[2][1] + landmarks[41][1]) // 2) + landmarks[41][1]) // 2

            x_75 = (((landmarks[14][0] + landmarks[41][0]) // 2) + landmarks[14][0]) // 2
            y_75 = (((landmarks[14][1] + landmarks[41][1]) // 2) + landmarks[14][1]) // 2

            x_76 = (((landmarks[14][0] + landmarks[41][0]) // 2) + landmarks[41][0]) // 2
            y_76 = (((landmarks[14][1] + landmarks[41][1]) // 2) + landmarks[41][1]) // 2

            x_77 = (((landmarks[3][0] + landmarks[43][0]) // 2) + landmarks[3][0]) // 2
            y_77 = (((landmarks[3][1] + landmarks[43][1]) // 2) + landmarks[3][1]) // 2

            x_78 = (((landmarks[13][0] + landmarks[47][0]) // 2) + landmarks[13][0]) // 2
            y_78 = (((landmarks[13][1] + landmarks[47][1]) // 2) + landmarks[13][1]) // 2

            new_landmark = [(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4), (x_5, y_5), (x_6, y_6),
                            (x_7, y_7), (x_8, y_8), (x_9, y_9), (x_10, y_10), (x_11, y_11), (x_12, y_12),
                            (x_13, y_13), (x_14, y_14), (x_15, y_15), (x_16, y_16), (x_17, y_17), (x_18, y_18),
                            (x_19, y_19), (x_20, y_20), (x_21, y_21), (x_22, y_22), (x_23, y_23), (x_24, y_24),
                            (x_25, y_25), (x_26, y_26), (x_27, y_27), (x_28, y_28), (x_29, y_29), (x_30, y_30),
                            (x_31, y_31), (x_32, y_32), (x_33, y_33), (x_34, y_34), (x_35, y_35), (x_36, y_36),
                            (x_37, y_37), (x_38, y_38), (x_39, y_39), (x_40, y_40), (x_41, y_41), (x_42, y_42),
                            (x_43, y_43), (x_44, y_44), (x_45, y_45), (x_46, y_46), (x_47, y_47), (x_48, y_48),
                            (x_49, y_49), (x_50, y_50), (x_51, y_51), (x_52, y_52), (x_53, y_53), (x_54, y_54),
                            (x_55, y_55), (x_56, y_56), (x_57, y_57), (x_58, y_58), (x_59, y_59), (x_60, y_60),
                            (x_61, y_61), (x_62, y_62), (x_63, y_63), (x_64, y_64), (x_65, y_65), (x_66, y_66),
                            (x_67, y_67), (x_68, y_68), (x_69, y_69), (x_70, y_70), (x_71, y_71), (x_72, y_72),
                            (x_73, y_73), (x_74, y_74), (x_75, y_75), (x_76, y_76), (x_77, y_77), (x_78, y_78)]

            for i in range(len(new_landmark)):
                x = new_landmark[i][0]
                y = new_landmark[i][1]

                if y - (y_spn // 20) < 0:
                    y = y_spn // 20
                if x - (x_spn // 20) < 0:
                    x = x_spn // 20

                if y - (y_spn // 20) >= img_raw.shape[0]:
                    y = img_raw.shape[0] + (y_spn // 20) - 1

                if x - (x_spn // 20) >= img_raw.shape[1]:
                    x = img_raw.shape[1] + (x_spn // 20) - 1

                node_map[i, k, 0] = int(np.mean(img_raw[y - (y_spn // 20): y + (y_spn // 20),
                                                x - (x_spn // 20): x + (x_spn // 20), 0]))
                node_map[i, k, 1] = int(np.mean(img_raw[y - (y_spn // 20): y + (y_spn // 20),
                                                x - (x_spn // 20): x + (x_spn // 20), 1]))
                node_map[i, k, 2] = int(np.mean(img_raw[y - (y_spn // 20): y + (y_spn // 20),
                                                x - (x_spn // 20): x + (x_spn // 20), 2]))

            img_face_ = img_raw[y_min: y_min + y_spn, x_min: x_min + x_spn]
            img_face = cv2.resize(img_face_, (7, 7))
            for i in range(49):
                h = i % 7
                w = i // 7
                node_map[i + 79, k, 0] = img_face[h, w, 0]
                node_map[i + 79, k, 1] = img_face[h, w, 1]
                node_map[i + 79, k, 2] = img_face[h, w, 2]

    node_map = cv2.resize(node_map, (rows_label - 1, 128))
    cv2.imwrite(learn_dataset[j] + 'node_map.png', node_map)
