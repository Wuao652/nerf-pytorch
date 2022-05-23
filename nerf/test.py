import numpy as np
import matplotlib.pyplot as plt

rootdir = '../cache/nerf_carla/UPPER_CAMERA/poses.npy'
upper_poses = np.load(rootdir)
upper_posi = upper_poses[:, :, -1]
rootdir = '../cache/nerf_carla/MIDDLE_CAMERA/poses.npy'
middle_poses = np.load(rootdir)
middle_posi = middle_poses[:, :, -1]
rootdir = '../cache/nerf_carla/LOWER_CAMERA/poses.npy'
lower_poses = np.load(rootdir)
lower_posi = lower_poses[:, :, -1]

posi = np.concatenate((upper_posi, middle_posi, lower_posi), 0)

upper_posi -= np.mean(posi, 0)
middle_posi -= np.mean(posi, 0)
lower_posi -= np.mean(posi, 0)
# upper_posi /= 15.0
# middle_posi /=15.0
# lower_posi /= 15.0
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(upper_posi[:, 0],
           upper_posi[:, 1],
           upper_posi[:, 2])
ax.scatter(middle_posi[:, 0],
           middle_posi[:, 1],
           middle_posi[:, 2])
ax.scatter(lower_posi[:, 0],
           lower_posi[:, 1],
           lower_posi[:, 2])
plt.show()