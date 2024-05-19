from PIL import Image
import numpy as np
import argparse
import random
import tqdm
import xlrd
import xlwt
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data, checkpoint
from torch.utils.data.dataset import T_co
from torchvision import transforms


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0, 1, 2, 3'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='vipl_v2', help='celeb, forensics, mahnob, vipl_v2')
parser.add_argument('--img_size', default=128, help='the height of input spatio-temporal map')
parser.add_argument('--window_size', default=300, help='the width of input spatio-temporal map')
parser.add_argument('--color_pattern', default='yuv', help='rgb, yuv')
parser.add_argument('--contrastive_learning', default=True, help='if use self-supervised contrast training')
parser.add_argument('--fine_tuning', default=False, helf='if use fine tuning')
parser.add_argument('--device', default='cuda')
parser.add_argument('--pretraining', default=True, helf='if use training weight')
parser.add_argument('--batch_size', default=10)
parser.add_argument('--learning_rate', default=1e-5)
parser.add_argument('--epochs', default=50)
parser.add_argument('--data_shuffle', default=True)
parser.add_argument('--save_img_dir', default='./save_dir/train_out_img/', helf='training process for visualization')
parser.add_argument('--save_weight', default=True, help='if save the training weight')
parser.add_argument('--save_weight_name', default='_pretraining_weight.pt', help='training weight file')
args = parser.parse_args()

img_size = args.img_size
window_size = args.window_size
batch_size = args.batch_size

if args.fine_tuning:
    assert args.dataset_name == 'vipl_V2'

if not os.path.exists(args.save_img_dir):
    os.makedirs(args.save_img_dir)


def img_norm(x):
    if np.max(x) == np.min(x):
        y = (x - np.min(x))
    else:
        y = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
    return y


def random_except(start, stop, excluded_value):
    while True:
        num = random.randint(start, stop)
        if num != excluded_value:
            return num


def split_last(x, shape):
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    s = x.size()
    assert n_dims > 1
    assert n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class DataAugmentation(object):
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        return self.transform(x)


class GenFineTuneDataset(data.Dataset):
    def __init__(self, img_names, img_shuffle_names, target_map_names, target_norm_names, target_ratios, trans):
        self.img_names = img_names
        self.img_shuffle_names = img_shuffle_names
        self.target_map_names = target_map_names
        self.target_norm_names = target_norm_names
        self.target_ratios = target_ratios
        self.trans = trans
        self.data_len = len(img_names)

    def __getitem__(self, index) -> T_co:
        k_1 = random_except(0, self.data_len - 1, index)
        k_2 = random_except(0, self.data_len - 1, index)

        img_norm_d = self.img_names[index]
        img_shuffle_d = self.img_shuffle_names[index]
        target_map_d = self.target_map_names[index]
        target_map_norm_d = self.target_norm_names[index]

        target_ratio = self.target_ratios[index]

        img_norm_d = Image.open(img_norm_d)
        img_norm_d = img_norm_d.convert('RGB')
        img_tensor = self.trans(img_norm_d)

        img_shuffle_d = Image.open(img_shuffle_d)
        img_shuffle_d = img_shuffle_d.convert('RGB')
        img_shuffle_tensor = self.trans(img_shuffle_d)

        target_map_d = Image.open(target_map_d)
        target_map_tensor = self.trans(target_map_d)

        target_norm_d = Image.open(target_map_norm_d)
        target_norm_tensor = self.trans(target_norm_d)

        target_w = target_map_d.resize((1, window_size))
        target_w = np.array(target_w)

        positive_2 = random.choice([self.img_names[index], self.img_shuffle_names[index]])
        positive_2 = Image.open(positive_2)
        positive_2 = positive_2.crop((0, 0, random.randint(1, img_size), window_size))
        positive_2 = positive_2.resize((img_size, window_size))
        positive_2 = positive_2.convert('RGB')
        positive_2_tensor = self.trans(positive_2)

        negative_1 = random.choice([self.img_names[k_1], self.img_shuffle_names[k_1]])

        negative_3 = Image.open(negative_1)
        negative_3 = negative_3.crop((0, 0, random.randint(1, img_size), window_size))
        negative_3 = negative_3.resize((img_size, window_size))
        negative_3 = negative_3.convert('RGB')
        negative_3_tensor = self.trans(negative_3)

        negative_1 = Image.open(negative_1)
        negative_1 = negative_1.convert('RGB')
        negative_1_tensor = self.trans(negative_1)

        negative_2 = random.choice([self.img_names[k_2], self.img_shuffle_names[k_2]])

        negative_4 = Image.open(negative_2)
        negative_4 = negative_4.crop((0, 0, random.randint(1, img_size), window_size))
        negative_4 = negative_4.resize((img_size, window_size))
        negative_4 = negative_4.convert('RGB')
        negative_4_tensor = self.trans(negative_4)

        negative_2 = Image.open(negative_2)
        negative_2 = negative_2.convert('RGB')
        negative_2_tensor = self.trans(negative_2)

        return img_tensor, img_shuffle_tensor, target_map_tensor, target_norm_tensor, target_ratio, target_w, \
               positive_2_tensor, negative_1_tensor, negative_2_tensor, negative_3_tensor, negative_4_tensor

    def __len__(self):
        return len(self.img_names)


class GenSelfSupervisedDataset(data.Dataset):
    def __init__(self, img_names, img_shuffle_names, trans):
        self.img_names = img_names
        self.img_shuffle_names = img_shuffle_names
        self.trans = trans
        self.data_len = len(img_names)

    def __getitem__(self, index) -> T_co:
        k_1 = random_except(0, self.data_len - 1, index)
        k_2 = random_except(0, self.data_len - 1, index)

        img_norm_d = self.img_names[index]
        img_shuffle_d = self.img_shuffle_names[index]

        img_norm_d = Image.open(img_norm_d)
        img_norm_d = img_norm_d.convert('RGB')
        img_tensor = self.trans(img_norm_d)

        img_shuffle_d = Image.open(img_shuffle_d)
        img_shuffle_d = img_shuffle_d.convert('RGB')
        img_shuffle_tensor = self.trans(img_shuffle_d)

        positive_2 = random.choice([self.img_names[index], self.img_shuffle_names[index]])
        positive_2 = Image.open(positive_2)
        positive_2 = positive_2.crop((0, 0, random.randint(1, img_size), window_size))
        positive_2 = positive_2.resize((img_size, window_size))
        positive_2 = positive_2.convert('RGB')
        positive_2_tensor = self.trans(positive_2)

        negative_1 = random.choice([self.img_names[k_1], self.img_shuffle_names[k_1]])

        negative_3 = Image.open(negative_1)
        negative_3 = negative_3.crop((0, 0, random.randint(1, img_size), window_size))
        negative_3 = negative_3.resize((img_size, window_size))
        negative_3 = negative_3.convert('RGB')
        negative_3_tensor = self.trans(negative_3)

        negative_1 = Image.open(negative_1)
        negative_1 = negative_1.convert('RGB')
        negative_1_tensor = self.trans(negative_1)

        negative_2 = random.choice([self.img_names[k_2], self.img_shuffle_names[k_2]])

        negative_4 = Image.open(negative_2)
        negative_4 = negative_4.crop((0, 0, random.randint(1, img_size), window_size))
        negative_4 = negative_4.resize((img_size, window_size))
        negative_4 = negative_4.convert('RGB')
        negative_4_tensor = self.trans(negative_4)

        negative_2 = Image.open(negative_2)
        negative_2 = negative_2.convert('RGB')
        negative_2_tensor = self.trans(negative_2)

        return img_tensor, img_shuffle_tensor,\
               positive_2_tensor, negative_1_tensor, negative_2_tensor, negative_3_tensor, negative_4_tensor

    def __len__(self):
        return len(self.img_names)


def loader_data():
    img_names = []
    img_shuffle_names = []

    root_dir = '../rppg_data/node_map_dataset/' + args.dataset_name + '/'

    if args.color_pattern == 'rgb':
        img_dir = root_dir + 'node_norm/'
    elif args.color_pattern == 'yuv':
        img_dir = root_dir + 'node_norm_yuv/'

    img_list = os.listdir(img_dir)
    img_list.sort()

    for img_0 in img_list:
        img_path = img_dir + img_0
        if img_path.endswith('.png'):
            img_names.append(img_path)

    if args.color_pattern == 'rgb':
        img_shuffle_dir = root_dir + 'node_shuffle/'
    elif args.color_pattern == 'yuv':
        img_shuffle_dir = root_dir + 'node_shuffle_yuv/'

    img_shf_list = os.listdir(img_shuffle_dir)
    img_shf_list.sort()

    for img_shf_0 in img_shf_list:
        img_shf_path = img_shuffle_dir + img_shf_0
        if img_shf_path.endswith('.png'):
            img_shuffle_names.append(img_shf_path)

    assert len(img_shuffle_names) == len(img_names)

    transform = DataAugmentation()

    if args.fine_tuning:
        target_map_names = []
        target_norm_names = []
        target_heart_rate = []

        root_list = os.listdir(root_dir)
        for file_0 in root_list:
            file_path = root_dir + file_0
            if file_path.endswith('hr_label.xls'):
                label_path = file_path
        label_file = xlrd.open_workbook(label_path)
        sheet = 'sheet'
        sheet_label = label_file.sheet_by_name(sheet)
        rows_label = sheet_label.nrows
        for i in range(rows_label):
            x = sheet_label.cell(i, 0).value
            target_heart_rate.append(x)
        target_heart_rate = target_heart_rate[1:]

        assert len(target_heart_rate) == len(img_names)

        target_ratios_matrix = []
        for i in range(len(target_heart_rate)):
            if target_heart_rate[i] > 135:
                target_heart_rate[i] = 135
            elif target_heart_rate[i] < 36:
                target_heart_rate[i] = 36
            target_ratio = np.zeros((100, 1), dtype=np.uint8)
            target_ratio[int(target_heart_rate[i] - 36)] = 1
            target_ratios_matrix.append(target_ratio)

        target_map_dir = root_dir + 'target_map/'
        target_map_list = os.listdir(target_map_dir)
        target_map_list.sort()
        for target_map_0 in target_map_list:
            target_map_path = target_map_dir + target_map_0
            if target_map_path.endswith('.png'):
                target_map_names.append(target_map_path)

        assert len(target_map_names) == len(img_names)

        target_norm_dir = root_dir + 'target_norm/'
        target_norm_list = os.listdir(target_norm_dir)
        target_norm_list.sort()
        for target_norm_0 in target_norm_list:
            target_norm_path = target_norm_dir + target_norm_0
            if target_norm_path.endswith('.png'):
                target_norm_names.append(target_norm_path)

        assert len(target_norm_names) == len(img_names)

        learn_ds = GenFineTuneDataset(img_names=img_names,
                                      img_shuffle_names=img_shuffle_names,
                                      target_map_names=target_map_names,
                                      target_norm_names=target_norm_names,
                                      target_ratios=target_ratios_matrix,
                                      trans=transform)

    else:
        learn_ds = GenSelfSupervisedDataset(img_names=img_names,
                                            img_shuffle_names=img_shuffle_names,
                                            trans=transform)

    learn_learning_data = data.DataLoader(dataset=learn_ds,
                                          batch_size=batch_size,
                                          shuffle=args.data_shuffle,
                                          drop_last=True)

    return learn_learning_data


learn_dl = loader_data()


class PreAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, frames, widths):
        super().__init__()
        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.Gap = nn.AvgPool2d(kernel_size=(1, widths))

        self.Fc1 = nn.Sequential(nn.Linear(frames, frames),
                                 nn.BatchNorm1d(out_channels),
                                 nn.ReLU(inplace=True),
                                 )

        self.Fc2 = nn.Sequential(nn.Linear(frames, frames),
                                 nn.BatchNorm1d(out_channels),
                                 nn.Sigmoid(),
                                 )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.Conv3 = nn.Conv2d(out_channels, 1, kernel_size=(1, 1))

    def forward(self, x):
        b, c, t, w = x.shape

        x = self.ConvBlock1(x)

        x_attn = self.Gap(x)
        x_attn = x_attn.flatten(2)
        x_attn = self.Fc1(x_attn)
        x_attn = self.Fc2(x_attn)
        x_attn = x_attn.view(b, c, t, 1)

        x = self.ConvBlock2(x)
        x = x * x_attn

        a_map = self.Conv3(x_attn)
        a_map.view(-1, t)
        return x, a_map


class PhysEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ConvBlockE1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockE2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockE3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockE4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockE5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockE6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockE7 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockE8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.DownSpa = nn.MaxPool2d((1, 2), stride=(1, 2))

        self.DownSpaTem = nn.MaxPool2d((2, 2), stride=(2, 2))

        self.Norm1 = nn.BatchNorm2d(64)
        self.Norm2 = nn.BatchNorm2d(128)
        self.Norm3 = nn.BatchNorm2d(256)
        self.Norm4 = nn.BatchNorm2d(512)

        self.Attn1 = PreAttentionBlock(in_channels=64, out_channels=64, frames=300, widths=128)
        self.Attn2 = PreAttentionBlock(in_channels=128, out_channels=128, frames=300, widths=64)
        self.Attn3 = PreAttentionBlock(in_channels=256, out_channels=256, frames=150, widths=32)
        self.Attn4 = PreAttentionBlock(in_channels=512, out_channels=512, frames=75, widths=16)

    def forward(self, x):
        b, c, t, w = x.shape

        x = self.ConvBlockE1(x)
        x_a1, a1 = self.Attn1(x)
        x = self.Norm1(x_a1 + x)
        x_128 = self.ConvBlockE2(x)

        x = self.DownSpa(x_128)
        x = self.ConvBlockE3(x)
        x_a2, a2 = self.Attn2(x)
        x = self.Norm2(x_a2 + x)
        x_64 = self.ConvBlockE4(x)

        x = self.DownSpaTem(x_64)
        x = self.ConvBlockE5(x)
        x_a3, a3 = self.Attn3(x)
        x = self.Norm3(x_a3 + x)
        x_32 = self.ConvBlockE6(x)

        x = self.DownSpaTem(x_32)
        x = self.ConvBlockE7(x)
        x_a4, a4 = self.Attn4(x)
        x = self.Norm4(x_a4 + x)
        x_16 = self.ConvBlockE8(x)

        x_8 = self.DownSpa(x_16)

        return x_8, x_16, x_32, x_64, x_128, a2


class PhysLatent(nn.Module):
    def __init__(self, frames, hr_classes):
        super().__init__()
        self.ConvBlockL1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockL2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.UpSpa = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )

        self.UpTem1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )

        self.UpTem2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )

        self.AvgPoolSpa = nn.AdaptiveAvgPool2d((frames, 1))

        self.ConvBlockW1 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(1),
        )

        self.AvgPoolSpaTem = nn.AdaptiveAvgPool2d((1, 1))

        self.ConvBlockH1 = nn.Sequential(nn.Conv2d(512, hr_classes, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                                         nn.BatchNorm2d(100),
                                         nn.Sigmoid(),
                                         )

    def forward(self, x):
        b, c, t, w = x.shape

        x = self.ConvBlockL1(x)
        x = self.ConvBlockL2(x)

        x_out = self.UpSpa(x)

        x_wave = self.UpTem1(x)
        x_wave = self.UpTem2(x_wave)
        x_wave = self.AvgPoolSpa(x_wave)
        x_wave = self.ConvBlockW1(x_wave)
        x_wave = x_wave.view(-1, 300)

        x_hr = self.AvgPoolSpaTem(x)
        x_hr = self.ConvBlockH1(x_hr)
        x_hr = x_hr.view(-1, 100)

        return x_out, x_wave, x_hr


class PhysDecoder(nn.Module):
    def __init__(self, out_channels, frames):
        super().__init__()

        self.ConvBlockD1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockD2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockD3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockD4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockD5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockD6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockD7 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockD8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlockD9 = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.UpSpaTem1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ELU(),
        )

        self.UpSpaTem2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
        )

        self.UpSpa = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        self.AvgPoolTem = nn.AdaptiveAvgPool2d((frames, 1))

    def forward(self, x, x_16, x_32, x_64, x_128):
        b, c, t, w = x.shape

        x = torch.cat((x, x_16), dim=1)
        x = self.ConvBlockD1(x)
        x = self.ConvBlockD2(x)

        x = self.UpSpaTem1(x)
        x = torch.cat((x, x_32), dim=1)
        x = self.ConvBlockD3(x)
        x = self.ConvBlockD4(x)

        x = self.UpSpaTem2(x)
        x = torch.cat((x, x_64), dim=1)
        x = self.ConvBlockD5(x)
        x = self.ConvBlockD6(x)

        x = self.UpSpa(x)
        x = torch.cat((x, x_128), dim=1)
        x = self.ConvBlockD7(x)
        x = self.ConvBlockD8(x)
        x = self.ConvBlockD9(x)

        x_wave = self.AvgPoolTem(x)
        x_wave = x_wave.view(-1, 300)

        return x, x_wave


class RefineAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Proj_q = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.BatchNorm2d(dim))

        self.Proj_k = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                    nn.BatchNorm2d(dim))

        self.Proj_v = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=0),
                                    nn.BatchNorm2d(dim))

        self.MLP = nn.Sequential(nn.Conv2d(dim, dim * 2, kernel_size=(1, 1)),
                                 nn.BatchNorm2d(dim * 2),
                                 nn.GELU(),
                                 nn.Conv2d(dim * 2,  dim * 2, kernel_size=(3, 3), stride=(1, 1), padding=1),
                                 nn.BatchNorm2d(dim * 2),
                                 nn.GELU(),
                                 nn.Conv2d(dim * 2, dim, kernel_size=(1, 1)),
                                 nn.BatchNorm2d(dim))

        self.n_heads = 8

        self.Norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        b, c, t, w = x.shape

        q = self.Proj_q(x)
        k = self.Proj_k(x)
        v = self.Proj_v(x)

        q = q.transpose(1, 2).flatten(2)
        k = k.transpose(1, 2).flatten(2)
        v = v.transpose(1, 2).flatten(2)

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        scores = q @ k.transpose(2, 3).contiguous()
        scores = F.softmax(scores, dim=-1)
        h = (scores @ v).transpose(1, 2).contiguous()

        h = merge_last(h, 2)

        h = h.view(b, t, c, w)
        h = h.transpose(2, 1).contiguous()
        x = x + h
        x = x + self.MLP(self.Norm(x))
        x = self.Norm(x)
        return x


class MainPhys(nn.Module):
    def __init__(self, in_channels, out_channels, frames, hr_classes):
        super().__init__()
        self.PhysEncoder = PhysEncoder(in_channels=in_channels)
        self.PhysLatent = PhysLatent(frames=frames, hr_classes=hr_classes)
        self.PhysDecoder = PhysDecoder(out_channels=out_channels, frames=window_size)

        self.Attn1 = RefineAttentionBlock(dim=512)
        self.Attn2 = RefineAttentionBlock(dim=256)
        self.Attn3 = RefineAttentionBlock(dim=128)
        self.Attn4 = RefineAttentionBlock(dim=64)

    def forward(self, x):
        b, c, t, w = x.shape
        x, x_16, x_32, x_64, x_128, attn_map = self.PhysEncoder(x)

        a_16 = self.Attn1(x_16)
        a_32 = self.Attn2(x_32)
        a_64 = self.Attn3(x_64)
        a_128 = self.Attn4(x_128)

        x, x_wave, x_hr = self.PhysLatent(x)
        x_map, gen_wave = self.PhysDecoder(x, a_16, a_32, a_64, a_128)
        return x_map, x_wave, x_hr, gen_wave, attn_map


class NegPearson(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self, xs, ys):
        loss = 0
        for i in range(xs.shape[0]):
            sum_x = torch.sum(xs[i])
            sum_y = torch.sum(ys[i])
            sum_xy = torch.sum(xs[i] * ys[i])
            sum_x2 = torch.sum(torch.pow(xs[i], 2))
            sum_y2 = torch.sum(torch.pow(ys[i], 2))
            N = ys.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
            loss += 1 - pearson

        loss = loss / ys.shape[0]
        return loss


class CalculateNormPSD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, :, 0] ** 2, x[:, :, 1] ** 2)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x


class FrequencyContrastiveLoss(nn.Module):
    def __init__(self, tau=0.08):
        super().__init__()
        self.norm_psd = CalculateNormPSD()
        self.distance_func = nn.MSELoss()
        self.tau = tau

    def forward(self, pos_waves, neg_waves):
        pos_dis_total = 0
        neg_dis_total = 0
        for i in range(len(pos_waves)):
            for j in range(len(pos_waves)):
                if i < j:
                    pos_f_i = self.norm_psd(pos_waves[i])
                    pos_f_i_norm = (pos_f_i - torch.min(pos_f_i)) / (torch.max(pos_f_i) - torch.min(pos_f_i))
                    pos_f_j = self.norm_psd(pos_waves[j])
                    pos_f_j_norm = (pos_f_j - torch.min(pos_f_j)) / (torch.max(pos_f_j) - torch.min(pos_f_j))
                    pos_dis = torch.exp(self.distance_func(pos_f_i_norm, pos_f_j_norm) / self.tau)
                    pos_dis_total += pos_dis

        for i in range(len(pos_waves)):
            for j in range(len(neg_waves)):
                pos_f = self.norm_psd(pos_waves[i])
                pos_f_norm = (pos_f - torch.min(pos_f)) / (torch.max(pos_f) - torch.min(pos_f))
                neg_f = self.norm_psd(neg_waves[j])
                neg_f_norm = (neg_f - torch.min(neg_f)) / (torch.max(neg_f) - torch.min(neg_f))
                neg_dis = torch.exp(self.distance_func(pos_f_norm, neg_f_norm) / self.tau)
                neg_dis_total += neg_dis

        loss = torch.log10((pos_dis_total / neg_dis_total) + 1)
        return loss


class WaveContrastiveLoss(nn.Module):
    def __init__(self, tau=0.08):
        super().__init__()
        self.distance_func = NegPearson()
        self.tau = tau

    def forward(self, pos_waves, neg_waves):
        pos_dis_total = 0
        neg_dis_total = 0
        for i in range(len(pos_waves)):
            for j in range(len(pos_waves)):
                if i < j:
                    pos_f_i = pos_waves[i]
                    pos_f_i_norm = (pos_f_i - torch.min(pos_f_i)) / (torch.max(pos_f_i) - torch.min(pos_f_i))
                    pos_f_j = pos_waves[j]
                    pos_f_j_norm = (pos_f_j - torch.min(pos_f_j)) / (torch.max(pos_f_j) - torch.min(pos_f_j))
                    pos_dis = torch.exp(self.distance_func(pos_f_i_norm, pos_f_j_norm) / self.tau)
                    pos_dis_total += pos_dis

        for i in range(len(pos_waves)):
            for j in range(len(neg_waves)):
                pos_f = pos_waves[i]
                pos_f_norm = (pos_f - torch.min(pos_f)) / (torch.max(pos_f) - torch.min(pos_f))
                neg_f = neg_waves[j]
                neg_f_norm = (neg_f - torch.min(neg_f)) / (torch.max(neg_f) - torch.min(neg_f))
                neg_dis = torch.exp(self.distance_func(pos_f_norm, neg_f_norm) / self.tau)
                neg_dis_total += neg_dis

        loss = torch.log10((pos_dis_total / neg_dis_total) + 1)
        return loss


def map_to_device(x, c):
    x = x.to(args.device)
    x = x.view(batch_size, c, window_size, img_size)
    x = (x - torch.mean(x)) / torch.std(x)
    return x


model = MainPhys(in_channels=3, out_channels=1, frames=300, hr_classes=100)


if torch.cuda.device_count() > 1:
    if torch.cuda.device_count() == 2:
        device_ids = [0, 1]
    elif torch.cuda.device_count() == 4:
        device_ids = [0, 1, 2, 3]
    model = nn.DataParallel(model, device_ids=device_ids)
    if args.pretraining:
        model_weight = './save_dir/check' + args.save_weight_name
        model.load_state_dict(torch.load(model_weight))
    model.to(args.device)
else:
    if args.pretraining:
        model_weight = './save_dir/check' + args.save_weight_name
        model.load_state_dict(torch.load(model_weight))
    model = model.cuda()

optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), eps=1e-8)

criterion_supervised_wave = NegPearson()
criterion_supervised_ratio = nn.L1Loss()
criterion_supervised_generative = nn.L1Loss()

criterion_contrastive_wave = WaveContrastiveLoss(tau=0.08)
criterion_contrastive_frequency = FrequencyContrastiveLoss(tau=0.08)

for epoch in range(args.epochs):
    learn_tqdm = tqdm.tqdm(iterable=learn_dl, total=len(learn_dl))
    learn_tqdm.set_description('learn epoch {:2d}'.format(epoch))

    for step, data in enumerate(learn_tqdm):
        if args.fine_tuning:
            imgs, img_shuffles, target_maps, target_norms, target_hrs, target_waves, pos_2s, neg_1s, neg_2s, neg_3s, neg_4s = data

            target_map = map_to_device(target_maps, c=1)
            target_norm = map_to_device(target_norms, c=1)

            target_hr = target_hrs.to(args.device)
            target_hr = target_hr.view(batch_size, 100)

            target_wave = target_waves.to(args.device)
            target_wave = target_wave.view(batch_size, window_size)
            target_wave = target_wave.float()
            target_wave = (target_wave - torch.mean(target_wave)) / torch.std(target_wave)

            target_img = random.choice([target_map, target_norm])

        else:
            imgs, img_shuffles, pos_2s, neg_1s, neg_2s, neg_3s, neg_4s = data

        img = map_to_device(imgs, c=3)
        img_shuffle = map_to_device(img_shuffles, c=3)
        in_img = random.choice([img, img_shuffle])

        pos_2 = map_to_device(pos_2s, c=3)
        neg_1 = map_to_device(neg_1s, c=3)
        neg_2 = map_to_device(neg_2s, c=3)
        neg_3 = map_to_device(neg_3s, c=3)
        neg_4 = map_to_device(neg_4s, c=3)

        pos_img_1, pos_wave_1_a, pos_hr_1, gen_wave, attn_map = model(in_img)
        _, pos_wave_2, _, _, _ = model(pos_2)
        _, neg_wave_1, _, _, _ = model(neg_1)
        _, neg_wave_2, _, _, _ = model(neg_2)
        _, neg_wave_3, _, _, _ = model(neg_3)
        _, neg_wave_4, _, _, _ = model(neg_4)

        pos_wave_1 = pos_wave_1_a.view(1, batch_size, window_size)
        pos_wave_2 = pos_wave_2.view(1, batch_size, window_size)
        neg_wave_1 = neg_wave_1.view(1, batch_size, window_size)
        neg_wave_2 = neg_wave_2.view(1, batch_size, window_size)
        neg_wave_3 = neg_wave_3.view(1, batch_size, window_size)
        neg_wave_4 = neg_wave_4.view(1, batch_size, window_size)

        pos_wave_array = torch.cat((pos_wave_1, pos_wave_2), dim=0)
        neg_wave_array = torch.cat((neg_wave_1, neg_wave_2, neg_wave_3, neg_wave_4), dim=0)

        c_loss_freq = criterion_contrastive_frequency(pos_wave_array, neg_wave_array)
        c_loss_wave = criterion_contrastive_wave(pos_wave_array, neg_wave_array)

        cl_freq_value = c_loss_freq.item()
        cl_wave_value = c_loss_wave.item()

        if args.fine_tuning:
            save_map = np.zeros((300, 300), dtype=np.uint8)

            attn_map_show = attn_map[0].cpu()
            attn_map_show = attn_map_show.detach().numpy()

            pos_wave_1_show = pos_wave_1_a[0].cpu()
            pos_wave_1_show = pos_wave_1_show.view(300, 1)
            pos_wave_1_show = pos_wave_1_show.detach().numpy()

            pos_img_out = pos_img_1[0].cpu()
            pos_img_out = pos_img_out.detach().numpy()

            target_img_out = target_img[0].cpu()
            target_img_out = target_img_out.detach().numpy()

            save_map[:, 0:128] = img_norm(pos_img_out)
            save_map[:, 128:256] = img_norm(target_img_out)
            save_map[:, 256:272] = img_norm(attn_map_show[:])
            save_map[:, 272:300] = img_norm(pos_wave_1_show[:])
            save_map = Image.fromarray(save_map)
            save_map = save_map.convert('RGB')
            save_map.save(args.save_img_dir + str(step).zfill(5) + '.png')

            s_loss_map = criterion_supervised_generative(pos_img_1, target_img)
            s_loss_wave = criterion_supervised_wave(pos_wave_1, target_wave)
            s_loss_hr = criterion_supervised_ratio(pos_hr_1, target_hr)

            sl_map_value = s_loss_map.item()
            sl_wave_value = s_loss_wave.item()
            sl_hr_value = s_loss_hr.item()

            s_loss_gen_wave = criterion_supervised_wave(gen_wave, target_wave)
            sl_gen_wave_value = s_loss_gen_wave.item()

            loss_total = s_loss_map + s_loss_wave + s_loss_hr + s_loss_gen_wave + c_loss_freq + c_loss_wave
            loss_total_value = loss_total.item()
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            learn_tqdm.desc = f'epoch [{epoch + 1}//{args.epochs}] l={loss_total_value:.5f}' \
                              f' map={sl_map_value:.5f} wave={sl_wave_value:.5f} hr={sl_hr_value:.5f}' \
                              f' c_freq={cl_freq_value:.5f} c_wave={cl_wave_value:.5f} g_wave={sl_wave_value:.5f}'

        else:
            loss_total = c_loss_freq + c_loss_wave
            loss_total_value = loss_total.item()
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            learn_tqdm.desc = f'epoch [{epoch + 1}//{args.epochs}] l={loss_total_value:.5f}' \
                              f' c_freq={cl_freq_value:.5f} c_wave={cl_wave_value:.5f}'

    if args.save_weight:
        torch.save(model.state_dict(), './save_dir/check' + args.save_weight_name)
