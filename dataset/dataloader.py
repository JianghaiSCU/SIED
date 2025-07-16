import os
import torch
import numpy as np
from PIL import Image as Image
import random
from torch.utils.data import Dataset


def remove_black_level(img, black_lv=512, white_lv=16383):
    img = np.maximum(img.astype(np.float32) - black_lv, 0) / (white_lv - black_lv)
    return img


def pack_raw(raw, black_level=512, white_level=16383):
    # pack Bayer image to 4 channels

    im = remove_black_level(raw, black_level, white_level)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


class Train_Dataset(Dataset):
    def __init__(self, image_dir, filelist, patch_size=(1024, 1024)):

        self.image_dir = image_dir

        self.file_list = os.path.join(self.image_dir, filelist)
        with open(self.file_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.patch_size = patch_size

        self.black_level = 512
        self.white_level = 16383

        self.image_width = 3840
        self.image_height = 2160

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')

        img_file, lbl_file = input_name.split(' ')

        low_RAW_img = np.fromfile(os.path.join(self.image_dir, img_file),
                                  dtype=np.uint16).reshape(self.image_height, self.image_width)
        
        low_RAW_img = pack_raw(low_RAW_img, black_level=self.black_level, white_level=self.white_level)

        high_RAW_img = np.fromfile(os.path.join(self.image_dir, lbl_file.replace('gt', 'gt_raw').replace('.jpg', '.raw')),
                                   dtype=np.uint16).reshape(self.image_height, self.image_width)
        
        high_RAW_img = pack_raw(high_RAW_img, black_level=self.black_level, white_level=self.white_level)
        
        high_RGB_img = np.asarray(Image.open(os.path.join(self.image_dir, lbl_file)))

        low_RAW_img, high_RAW_img, high_RGB_img = self.random_crop(low_RAW_img, high_RAW_img, high_RGB_img)

        data = {"low_RAW_img": low_RAW_img, "high_RAW_img": high_RAW_img, "high_RGB_img": high_RGB_img}

        return data

    def random_crop(self, img1, img2, img3, start=None, patch_size=None):

        height, width = img1.shape[:2]

        if patch_size is None:
            patch_size_h, patch_size_w = self.patch_size
        else:
            patch_size_h, patch_size_w = patch_size

        if start is None:
            x = random.randint(0, width - patch_size_w - 1)
            y = random.randint(0, height - patch_size_h - 1)
        else:
            x, y = start
        img1_patch = img1[y: y + patch_size_h, x: x + patch_size_w, :]
        img2_patch = img2[y: y + patch_size_h, x: x + patch_size_w, :]
        img3_patch = img3[y * 2: y * 2 + patch_size_h * 2, x * 2: x * 2 + patch_size_w * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            img1_patch = np.flip(img1_patch, axis=1)
            img2_patch = np.flip(img2_patch, axis=1)
            img3_patch = np.flip(img3_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            img1_patch = np.flip(img1_patch, axis=0)
            img2_patch = np.flip(img2_patch, axis=0)
            img3_patch = np.flip(img3_patch, axis=0)
        
        img1_patch, img2_patch, img3_patch = torch.tensor(img1_patch.copy()).permute(2, 0, 1).float(), \
            torch.tensor(img2_patch.copy()).permute(2, 0, 1).float(), \
                torch.tensor(img3_patch.copy() / 255.0).permute(2, 0, 1).float()

        return img1_patch, img2_patch, img3_patch

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)


class Test_Dataset(Dataset):
    def __init__(self, image_dir, filelist):
        self.image_dir = image_dir

        self.file_list = os.path.join(self.image_dir, filelist)
        with open(self.file_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names

        self.black_level = 512
        self.white_level = 16383

        self.image_width = 3840
        self.image_height = 2160

    def get_images(self, index):
        input_name = self.input_names[index].replace('\n', '')

        img_file, lbl_file = input_name.split(' ')
        
        img_name = img_file.split('/')[-1].replace('.raw', '.jpg')

        low_RAW_img = np.fromfile(os.path.join(self.image_dir, img_file),
                                  dtype=np.uint16).reshape(self.image_height, self.image_width)
        
        low_RAW_img = pack_raw(low_RAW_img, black_level=self.black_level, white_level=self.white_level)

        high_RAW_img = np.fromfile(os.path.join(self.image_dir, lbl_file.replace('gt', 'gt_raw').replace('.jpg', '.raw')),
                                   dtype=np.uint16).reshape(self.image_height, self.image_width)
        
        high_RAW_img = pack_raw(high_RAW_img, black_level=self.black_level, white_level=self.white_level)
        
        high_RGB_img = np.asarray(Image.open(os.path.join(self.image_dir, lbl_file)))

        low_RAW_img, high_RAW_img, high_RGB_img = torch.tensor(low_RAW_img).permute(2, 0, 1).float(), \
            torch.tensor(high_RAW_img).permute(2, 0, 1).float(), \
                torch.tensor(high_RGB_img / 255.0).permute(2, 0, 1).float()

        data = {"low_RAW_img": low_RAW_img, "high_RAW_img": high_RAW_img, "high_RGB_img": high_RGB_img, "img_name": img_name}

        return data

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)