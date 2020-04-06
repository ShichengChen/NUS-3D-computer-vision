""" CS4277/CS5477 Lab 4: Plane Sweep Stereo
See accompanying Jupyter notebook (lab4.ipynb) for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
NUSNET ID: e1234567

Name2: <Name of second member, if any>
Email2: <username>@u.nus.edu
NUSNET ID2: e1234567
"""
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import scipy.ndimage

"""Helper functions: You should not have to touch the following functions.
"""


class Image(object):
    """
    Image class. You might find the following member variables useful:
    - image: RGB image (HxWx3) of dtype np.float64
    - pose_mat: 3x4 Camera extrinsics that transforms points from world to
        camera frame
    """

    def __init__(self, qvec, tvec, name, root_folder=''):
        self.qvec = qvec
        self.tvec = tvec
        self.name = name  # image filename
        self._image = self.load_image(os.path.join(root_folder, name))

        # Extrinsic matrix: Transforms from world to camera frame
        self.pose_mat = self.make_extrinsic(qvec, tvec)

    def __repr__(self):
        return '{}: qvec={}\n tvec={}'.format(
            self.name, self.qvec, self.tvec
        )

    @property
    def image(self):
        return self._image.copy()

    @staticmethod
    def load_image(path):
        """Loads image and converts it to float64"""
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im.astype(np.float64) / 255.0

    @staticmethod
    def make_extrinsic(qvec, tvec):
        """ Make 3x4 camera extrinsic matrix from colmap pose

        Args:
            qvec: Quaternion as per colmap format (q_cv) in the order
                  q_w, q_x, q_y, q_z
            tvec: translation as per colmap format (t_cv)

        Returns:

        """
        rotation = Rotation.from_quat(np.roll(qvec, -1))
        return np.concatenate([rotation.as_dcm(), tvec[:, None]], axis=1)


def write_json(outfile, images, intrinsic_matrix, img_hw):
    """Write metadata to json file.

    Args:
        outfile (str): File to write to
        images (list): List of Images
        intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix
        img_hw (tuple): (image height, image width)
    """

    img_height, img_width = img_hw

    images_meta = []
    for im in images:
        images_meta.append({
            'name': im.name,
            'qvec': im.qvec.tolist(),
            'tvec': im.tvec.tolist(),
        })

    data = {
        'img_height': img_height,
        'img_width': img_width,
        'K': intrinsic_matrix.tolist(),
        'images': images_meta
    }
    with open(outfile, 'w') as fid:
        json.dump(data, fid, indent=2)


def load_data(root_folder):
    """Loads dataset.

    Args:
        root_folder (str): Path to data folder. Should contain metadata.json

    Returns:
        images, K, img_hw
    """
    print('Loading data from {}...'.format(root_folder))
    with open(os.path.join(root_folder, 'metadata.json')) as fid:
        metadata = json.load(fid)

    images = []
    for im in metadata['images']:
        images.append(Image(np.array(im['qvec']), np.array(im['tvec']),
                            im['name'], root_folder=root_folder))
    img_hw = (metadata['img_height'], metadata['img_width'])
    K = np.array(metadata['K'])

    print('Loaded data containing {} images.'.format(len(images)))
    return images, K, img_hw


def rgb2hex(rgb):
    """Converts color representation into hexadecimal representation for K3D

    Args:
        rgb (np.ndarray): (N, 3) array holding colors

    Returns:
        hex (np.ndarray): array (N, ) of size N, each element indicates the
          color, e.g. 0x0000FF = blue
    """
    rgb_uint = (rgb * 255).astype(np.uint8)
    hex = np.sum(rgb_uint * np.array([[256 ** 2, 256, 1]]),
                 axis=1).astype(np.uint32)
    return hex


"""Functions to be implemented
"""


# Part 1
def compute_relative_pose(cam_pose, ref_pose):
    """Compute relative pose between two cameras

     Args:
        cam_pose (np.ndarray): Extrinsic matrix of camera of interest C_i (3,4).
          Transforms points in world frame to camera frame, i.e.
            x_i = C_i @ x_w  (taking into account homogeneous dimensions)
        ref_pose (np.ndarray): Extrinsic matrix of reference camera C_r (3,4)

    Returns:
        relative_pose (np.ndarray): Relative pose of size (3,4). Should transform 
          points in C_r to C_i, i.e. x_i = M @ x_r

    Prohibited functions:
        Do NOT use np.linalg.inv() or similar functions
    """
    relative_pose = np.zeros((3, 4), dtype=np.float64)

    """ YOUR CODE STARTS HERE """

    Ri, Rr = cam_pose[:, :-1], ref_pose[:, :-1]
    ti, tr = cam_pose[:, -1:], ref_pose[:, -1:]
    relative_pose[:, :-1] = Ri @ Rr.T
    relative_pose[:, -1:] = ti - Ri @ Rr.T @ tr

    """ YOUR CODE ENDS HERE """
    return relative_pose


def get_plane_sweep_homographies(K, relative_pose, inv_depths):
    """Compute plane sweep homographies, assuming fronto parallel planes w.r.t.
    reference camera

    Args:
        K (np.ndarray): Camera intrinsic matrix (3,3)
        relative_pose (np.ndarray): Relative pose between the two cameras
          of shape (3, 4)
        inv_depths (np.ndarray): Inverse depths to warp of size (D, )

    Returns:
        homographies (D, 3, 3)
    """

    homographies = None

    """ YOUR CODE STARTS HERE """
    r = relative_pose[:, :-1]
    t = relative_pose[:, -1:]
    n = np.array([[0, 0, 1]])
    invk = np.linalg.inv(K)
    le = np.expand_dims(K @ r @ invk, axis=0)
    inv_depths = np.expand_dims(inv_depths, axis=(1, 2))
    homographies = K @ ((t @ n) * inv_depths) @ invk + le

    """ YOUR CODE ENDS HERE """

    return homographies


# Part 2
def compute_plane_sweep_volume(images, ref_pose, K, inv_depths, img_hw):
    """Compute plane sweep volume, by warping all images to the reference camera
    fronto-parallel planes, before computing the variance for each pixel and
    depth.

    Args:
        images (list[Image]): List of images which contains information about
          the camera extrinsics for each image
        ref_pose (np.ndarray): Reference camera pose
        K (np.ndarray): 3x3 intrinsic matrix (assumed same for all cameras)
        inv_depths (list): List of inverse depths to consider for plane sweep
        img_hw (tuple): tuple containing (H, W), which are the output height
          and width for the plane sweep volume.

    Returns:
        ps_volume (np.ndarray):
          Plane sweep volume of size (D, H, W), with dtype=np.float64, where
          D is len(inv_depths), and (H, W) are the image heights and width
          respectively. Each element should contain the variance of all pixel
          intensities that warp onto it.
        accum_count (np.ndarray):
          Accumulator count of same size as ps_volume, and dtype=np.int32.
          Keeps track of how many images are warped into a certain pixel,
          i.e. the number of pixels used to compute the variance.
        extras (any type):
          Any additional information you might want to keep for part 4.
    """

    D = len(inv_depths)
    H, W = img_hw
    N = len(images)
    MD = 64
    ps_volume = np.zeros((D, H, W), dtype=np.float32)
    ps_all = np.zeros([N, MD, H, W, 3])
    accum_count = np.zeros((D, H, W), dtype=np.uint8)
    mask = np.zeros([N, MD, H, W, 3], dtype=np.uint8)
    extras = []

    relative_pose = np.zeros([N, 3, 4])
    H = np.zeros([N, D, 3, 3])

    def wvar(a, w):
        av = np.average(a, weights=w, axis=0)
        av = np.expand_dims(av, axis=0)
        var = np.average((a - av) ** 2, weights=w, axis=0)
        return var

    def showimage(imgstr, img):
        cv2.imshow(imgstr, np.clip(img, 0, 255))

    for i in range(N):
        relative_pose[i] = compute_relative_pose(images[i].pose_mat, ref_pose)
        H[i] = get_plane_sweep_homographies(K, relative_pose[i], np.array(inv_depths))
        # for j in range(D):
        #     img = cv2.warpPerspective(images[i].image, H[i][j], img_hw[::-1],borderValue=-1)
        # showimage('img', np.clip(img,0,255))
        # showimage('img',img)
        # cv2.waitKey(0)

    for j in range(0, D):
        for i in range(N):
            ps_all[i][j % MD] = cv2.warpPerspective(images[i].image, H[i][j], img_hw[::-1], borderValue=-1)
            # print(ps_all[i])
            # showimage('img', ps_all[i])
            # cv2.waitKey(0)
        if (j % MD == MD - 1):  # mask=np.zeros([N,MD,H,W,3],dtype=np.uint8)
            mask[:, :, :, :, :] = (ps_all != -1)
            accum_count[j - MD + 1:j + 1] += np.sum(mask[:, :, :, :, 0], axis=0)
            ps_volume[j - MD + 1:j + 1] = np.mean(wvar(ps_all, w=mask), axis=-1)
            print(j, i)

    # cv2.destroyAllWindows()
    """ YOUR CODE STARTS HERE """

    """ YOUR CODE ENDS HERE """

    return ps_volume, accum_count, extras


def compute_depths(ps_volume, inv_depths):
    """Computes inverse depth map from plane sweep volume as the
    argmin over plane sweep volume variances.

    Args:
        ps_volume (np.ndarray): Plane sweep volume of size (D, H, W) from
          compute_plane_sweep_volume()
        inv_depths (np.ndarray): List of depths considered in the plane
          sweeping (D,)

    Returns:
        inv_depth_image (np.ndarray): inverse-depth estimate (H, W)
    """
    D, H, W = ps_volume.shape
    # inv_depth_image = np.zeros(ps_volume.shape[1:], dtype=np.float64)
    # inv_depths_img=np.repeat(inv_depths.reshape(D,1),H*W,axis=1).reshape(D,H,W)

    """ YOUR CODE STARTS HERE """
    idx = np.argmin(ps_volume, axis=0)
    print(idx)
    inv_depth_image = inv_depths[idx]
    """ YOUR CODE ENDS HERE """

    return inv_depth_image


# Part 3
def unproject_depth_map(image, inv_depth_image, K, mask=None):
    """Converts the depth map into points by unprojecting depth map into 3D

    Note: You will also need to implement the case where no mask is provided

    Args:
        image (np.ndarray): Image bitmap (H, W, 3)
        inv_depth_image (np.ndarray): Inverse depth image (H, W)
        K (np.ndarray): 3x3 Camera intrinsics
        mask (np.ndarray): Optional mask of size (H, W) and dtype=np.bool.

    Returns:
        xyz (np.ndarray): Nx3 coordinates of points, dtype=np.float64.
        rgb (np.ndarray): Nx3 RGB colors, where rgb[i, :] is the (Red,Green,Blue)
          colors for the points at position xyz[i, :]. Should be in the range
          [0, 1] and have dtype=np.float64.
    """

    xyz = np.zeros([0, 3], dtype=np.float64)
    rgb = np.zeros([0, 3], dtype=np.float64)  # values should be within (0, 1)

    """ YOUR CODE STARTS HERE """

    """ YOUR CODE ENDS HERE """

    return xyz, rgb


# Part 4
def post_process(ps_volume, inv_depth, accum_count, extras):
    """Post processes the plane sweep volume and compute a mask to indicate
    which pixels have confident estimates of the depth

    Args:
        ps_volume: Plane sweep volume from compute_plane_sweep_volume()
          of size (D, H, W)
        inv_depths (List[float]): List of depths considered in the plane
          sweeping
        accum_count: Accumulator count from compute_plane_sweep_volume(), which
          can be used to indicate which pixels are not observed by many other
          images.
        extras: Extra variables from compute_plane_sweep_volume() in Part 2

    Returns:
        inv_depth_image: Denoised Inverse depth image (similar to compute_depths)
        mask: np.ndarray of size (H, W) and dtype np.bool.
          Pixels with values TRUE indicate valid pixels.
    """

    mask = np.ones(ps_volume.shape[1:], dtype=np.bool)
    inv_depth_filtered = inv_depth.copy()

    """ YOUR CODE STARTS HERE """

    """ YOUR CODE ENDS HERE """

    return inv_depth_filtered, mask
