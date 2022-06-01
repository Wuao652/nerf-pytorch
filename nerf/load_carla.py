import os
import sys

import numpy as np
import imageio
import cv2
import matplotlib.pyplot as plt

def load_carla_data(basedir,
                    downsample_factor=2.0,
                    face_forward=False,
                    high_resolution=False):
    """
    This is the function to load and pre-process the rgb images.
    Args:
        basedir: str. the path to store the carla rgb images.
        downsample_factor: float or None. a factor to downsample the input rgb-images.
                           If set to None, it means directly loading the raw images.
    Returns:
        images : numpy array [N, H, W, 3]
        poses : numpy array [N, 3, 4]
        hwf : list of 3 elements [H, W, focal]
        i_split : list of 3 list. the index of train, validate and test images
    """
    print('start load carla data!')
    if high_resolution:
        cameralocs = ['upper2560', 'middle2560', 'lower2560']
        H, W, focal = 1440, 2560, 896.2656489084286
    else:
        cameralocs = ['upper640', 'middle640', 'lower640']
        H, W, focal = 480, 640, 224.06641222710715
        downsample_factor = None


    if downsample_factor:
        H /= downsample_factor
        W /= downsample_factor
        focal /= downsample_factor
        H, W = int(H), int(W)

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

        def imread(f):
            if f.endswith("png") or f.endswith("PNG"):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)

        if downsample_factor:
            # TODO: resize images using INTER_AREA (cv2)
            rgb_image = [
                cv2.resize(imread(f)[..., :3] / 255.0,
                           dsize=(W, H),
                           interpolation=cv2.INTER_AREA)
                for f in rgb_img]
        else:
            rgb_image = [
                imread(f)[..., :3] / 255.0
                for f in rgb_img]

        rgb_image = np.stack(rgb_image, 0)
        rgb_image = rgb_image.astype(np.float32)
        rgb_image = rgb_image[idx]

        pose = np.load(os.path.join(basedir, c, 'poses.npy'))
        pose = pose.astype(np.float32)

        if face_forward:
            rgb_image = rgb_image[56:71]
            pose = pose[56:71]
        images.append(rgb_image)
        poses.append(pose)

    images = np.concatenate(images, axis=0)
    images = images.astype(np.float32)
    print(images.shape)
    # if half_res:
    #     # TODO: resize images using INTER_AREA (cv2)
    #     H = H // 2
    #     W = W // 2
    #     focal = focal / 2.0
    #     images = [
    #         cv2.resize(images[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
    #         for i in range(images.shape[0])
    #     ]
    #     images = np.stack(images, 0)
    poses = np.concatenate(poses, axis=0)
    poses = poses.astype(np.float32)

    if not face_forward:
    # shift only x and y
        xy = poses[:, :-1, -1]
        poses[:, :-1, -1] = poses[:, :-1, -1] - np.mean(xy, 0)
        # xyz = poses[..., -1]
        # scale_factor = np.max(np.fabs(xyz))
        scale_factor = 6.0
        print('scale_facter: ', scale_factor)
        poses[:, :, -1] /= scale_factor
    else:
        scale_factor = 6.0
        xy_mean = np.array([-52.07071, -230.9998], dtype=np.float32)
        poses[:, :-1, -1] = poses[:, :-1, -1] - xy_mean
        poses[:, :, -1] /= scale_factor

    poses[:, :, 1] = - poses[:, :, 1]
    poses[:, :, 2] = - poses[:, :, 2]

    np.random.seed(34)
    hwf = [H, W, focal]

    i_choice = np.random.choice(images.shape[0], images.shape[0], False)
    i_split = [list(range(i, i+images.shape[0] // 3)) for i in range(0, images.shape[0], images.shape[0] // 3)]
    i_split = [i_choice[s] for s in i_split]

    i_split[1] = np.array([i_split[1][0]])

    print('images: ', images.shape)
    print('poses: ', poses.shape)
    print('hwf', hwf)
    print('i_split', i_split)
    return images.astype(np.float32), poses.astype(np.float32), hwf, i_split

if __name__ =='__main__':
    basedir = '../cache/nerf_carla_new'
    images, poses, hwf, i_split = load_carla_data(basedir)

