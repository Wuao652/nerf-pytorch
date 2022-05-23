import os
import sys

import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt

def load_carla_data(basedir, half_res=True):
    print('start load carla data!')
    cameralocs = ['UPPER_CAMERA', 'MIDDLE_CAMERA', 'LOWER_CAMERA']
    H, W, focal = 1080, 1920, 672.1992366813214
    poses = []
    images = []
    for c in cameralocs:
        img_list = os.listdir(os.path.join(basedir, c))
        if 'poses.npy' in img_list:
            img_list.remove('poses.npy')
        idx_list = np.array([int(i[:-4][2:]) for i in img_list])
        idx = np.argsort(idx_list)

        rgb_img = [
            os.path.join(basedir, c, f)
            for f in img_list
            if f.endswith("png") or f.endswith("PNG")
        ]

        # img0 = rgb_img[0]
        # H, W, _ = imageio.imread(img0).shape
        def imread(f):
            if f.endswith("png") or f.endswith("PNG"):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)


        rgb_image = [imread(f)[..., :3] / 255.0 for f in rgb_img]
        rgb_image = np.stack(rgb_image, 0)
        rgb_image = rgb_image.astype(np.float32)
        rgb_image = rgb_image[idx]
        images.append(rgb_image)

        pose = np.load(os.path.join(basedir, c, 'poses.npy'))
        pose = pose.astype(np.float32)
        poses.append(pose)

    images = np.concatenate(images, axis=0)
    images = images.astype(np.float32)
    print(images.shape)
    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = H // 2
        W = W // 2
        focal = focal / 2.0
        images = [
            cv2.resize(images[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
            for i in range(images.shape[0])
        ]
        images = np.stack(images, 0)
    poses = np.concatenate(poses, axis=0)
    poses = poses.astype(np.float32)

    # shift only x and y
    xy = poses[:, :-1, -1]

    poses[:, :-1, -1] = poses[:, :-1, -1] - np.mean(xy, 0)
    poses[:, :, -1] /= 18.0

    hwf = [H, W, focal]
    i_split = [list(range(i, i+39)) for i in range(0, images.shape[0], 39)]
    i_split[1] = [i_split[0][0]]

    print('images: ', images.shape)
    print('poses: ', poses.shape)
    print('hwf', hwf)
    print('i_split', i_split)
    return images, poses, hwf, i_split

if __name__ =='__main__':
    basedir = '../cache/nerf_carla'
    images, poses, hwf, i_split = load_carla_data(basedir)

