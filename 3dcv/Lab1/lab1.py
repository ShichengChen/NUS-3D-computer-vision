""" CS4277/CS5477 Lab 1: Fun with Homographies.
See accompanying Jupyter notebook (lab1.ipynb) for instructions.

Name: Shicheng Chen
Email: e0534721@u.nus.edu
Student ID: A0215003A

Name2: <Name of second member, if any>
Email2: <username>@u.nus.edu
Student ID: A0123456X
"""
from math import floor, ceil, sqrt

import cv2
import numpy as np

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)


"""Helper functions: You should not have to touch the following functions.
"""
def load_image(im_path):
    """Loads image and converts to RGB format

    Args:
        im_path (str): Path to image

    Returns:
        im (np.ndarray): Loaded image (H, W, 3), of type np.uint8.
    """
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def draw_matches(im1, im2, im1_pts, im2_pts, inlier_mask=None):
    """Generates a image line correspondences

    Args:
        im1 (np.ndarray): Image 1
        im2 (np.ndarray): Image 2
        im1_pts (np.ndarray): Nx2 array containing points in image 1
        im2_pts (np.ndarray): Nx2 array containing corresponding points in
          image 2
        inlier_mask (np.ndarray): If provided, inlier correspondences marked
          with True will be drawn in green, others will be in red.

    Returns:

    """
    height1, width1 = im1.shape[:2]
    height2, width2 = im2.shape[:2]
    canvas_height = max(height1, height2)
    canvas_width = width1 + width2

    canvas = np.zeros((canvas_height, canvas_width, 3), im1.dtype)
    canvas[:height1, :width1, :] = im1
    canvas[:height2, width1:width1+width2, :] = im2

    im2_pts_adj = im2_pts.copy()
    im2_pts_adj[:, 0] += width1

    if inlier_mask is None:
        inlier_mask = np.ones(im1_pts.shape[0], dtype=np.bool)

    # Converts all to integer for plotting
    im1_pts = im1_pts.astype(np.int32)
    im2_pts_adj = im2_pts_adj.astype(np.int32)

    # Draw points
    all_pts = np.concatenate([im1_pts, im2_pts_adj], axis=0)
    for pt in all_pts:
        cv2.circle(canvas, (pt[0], pt[1]), 4, _COLOR_BLUE, 2)

    # Draw lines
    for i in range(im1_pts.shape[0]):
        pt1 = tuple(im1_pts[i, :])
        pt2 = tuple(im2_pts_adj[i, :])
        color = _COLOR_GREEN if inlier_mask[i] else _COLOR_RED
        cv2.line(canvas, pt1, pt2, color, 2)

    return canvas


def matches2pairs(matches, kp1, kp2):
    """Converts OpenCV's DMatch to point pairs

    Args:
        matches (list): List of DMatches from OpenCV's matcher
        kp1 (list): List of cv2.KeyPoint from OpenCV's detector for image 1 (query)
        kp2 (list): List of cv2.KeyPoint from OpenCV's detector for image 2 (train)

    Returns:
        pts1, pts2: Nx2 array containing corresponding coordinates for both images
    """

    pts1, pts2 = [], []
    for m in matches:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.stack(pts1, axis=0)
    pts2 = np.stack(pts2, axis=0)

    return pts1, pts2


"""Functions to be implemented
"""

# Part 1(a)
def transform_homography(src, h_matrix):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    Prohibited functions:
        cv2.perspectiveTransform()

    """
    #print('src',src.shape)
    #print('np.ones(src.shape[0],1))',np.ones((src.shape[0],1)).shape)
    home = np.concatenate((src,np.ones((src.shape[0],1))),axis=1)
    transformed = np.matmul(h_matrix,home.T).T
    transformed=transformed[:,:2]/transformed[:,2:3]
    """ YOUR CODE STARTS HERE """

    """ YOUR CODE ENDS HERE """

    return transformed


def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """
    N=src.shape[0]
    c0=np.array([[src[:,0].sum()/N,src[:,1].sum()/N]])
    c1=np.array([[dst[:,0].sum()/N,dst[:,1].sum()/N]])
    s0=np.sqrt(2)/np.sum(np.sqrt((src-c0)**2).sum()/N)
    s1=np.sqrt(2)/np.sum(np.sqrt((dst-c1)**2).sum()/N)
    N0=np.array([[s0,0,-s0*c0[0,0]],[0,s0,-s0*c0[0,1]],[0,0,1]])
    N1=np.array([[s1,0,-s1*c1[0,0]],[0,s1,-s1*c1[0,1]],[0,0,1]])
    src=transform_homography(src,N0)
    dst=transform_homography(dst,N1)
    arr = []
    for i in range(src.shape[0]):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        arr.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        arr.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    arr = np.asarray(arr)
    U, S, Vh = np.linalg.svd(arr)
    L = Vh[-1, :]
    H = L.reshape(3, 3)
    H=np.matmul(np.matmul(np.linalg.inv(N1),H),N0)
    H = H / H[-1, -1]


    """ YOUR CODE STARTS HERE """
    
    """ YOUR CODE ENDS HERE """

    return H


# Part 2
def warp_image(src, dst, h_matrix):
    """Applies perspective transformation to source image to warp it onto the
    destination (background) image

    Args:
        src (np.ndarray): Source image to be warped
        dst (np.ndarray): Background image to warp template onto
        h_matrix (np.ndarray): Warps coordinates from src to the dst, i.e.
                                 x_{dst} = h_matrix * x_{src},
                               where x_{src}, x_{dst} are the homogeneous
                               coordinates in I_{src} and I_{dst} respectively

    Returns:
        dst (np.ndarray): Source image warped onto destination image

    Prohibited functions:
        cv2.warpPerspective()
    """
    dst =dst.copy() 
    """ YOUR CODE STARTS HERE """
    H = np.linalg.inv(h_matrix)
    h, w = dst.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xy = np.array([xx.ravel(), yy.ravel()]).astype(np.float32).T
    xy=transform_homography(xy,H)
    map_x = xy[:,0].reshape(h, w).astype(np.float32)
    map_y = xy[:,1].reshape(h, w).astype(np.float32)

    cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, dst=dst, borderMode=cv2.BORDER_TRANSPARENT)
    """ YOUR CODE ENDS HERE """

    return dst


def warp_images_all(images, h_matrices):
    """Warps all images onto a black canvas.

    Note: We implemented this function for you, but it'll be useful to note
     the necessary steps
     1. Compute the bounds of each of the images (which can be negative)
     2. Computes the necessary size of the canvas
     3. Adjust all the homography matrices to the canvas bounds
     4. Warp images

    Requires:
        transform_homography(), warp_image()

    Args:
        images (List[np.ndarray]): List of images to warp
        h_matrices (List[np.ndarray]): List of homography matrices

    Returns:
        stitched (np.ndarray): Stitched images
    """
    assert len(images) == len(h_matrices) and len(images) > 0
    num_images = len(images)

    corners_transformed = []
    for i in range(num_images):
        h, w = images[i].shape[:2]
        bounds = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
        transformed_bounds = transform_homography(bounds, h_matrices[i])
        corners_transformed.append(transformed_bounds)
    corners_transformed = np.concatenate(corners_transformed, axis=0)

    # Compute required canvas size
    min_x, min_y = np.min(corners_transformed, axis=0)
    max_x, max_y = np.max(corners_transformed, axis=0)
    min_x, min_y = floor(min_x), floor(min_y)
    max_x, max_y = ceil(max_x), ceil(max_y)

    canvas = np.zeros((max_y-min_y, max_x-min_x, 3), images[0].dtype)

    for i in range(num_images):
        # adjust homography matrices
        trans_mat = np.array([[1.0, 0.0, -min_x],
                              [0.0, 1.0, -min_y],
                              [0.0, 0.0, 1.0]], h_matrices[i].dtype)
        h_adjusted = trans_mat @ h_matrices[i]

        # Warp
        canvas = warp_image(images[i], canvas, h_adjusted)

    return canvas


# Part 3
def compute_homography_error(src, dst, homography):
    """Compute the squared bidirectional pixel reprojection error for
    provided correspondences

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        homography (np.ndarray): Homography matrix that transforms src to dst.

    Returns:
        err (np.ndarray): Array of size (N, ) containing the error d for each
        correspondence, computed as:
          d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
        where ||a|| denotes the l2 norm (euclidean distance) of vector a.
    """
    invH=np.linalg.inv(homography)
    H=homography
    #print(src.shape)
    a=np.sum((transform_homography(src,H)-dst)**2,axis=1)
    b=np.sum((transform_homography(dst,invH)-src)**2,axis=1)
    """ YOUR CODE STARTS HERE """
    
    """ YOUR CODE ENDS HERE """

    return a+b


def compute_homography_ransac(src, dst, thresh=16.0, num_tries=200):
    """Calculates the perspective transform from at least 4 points of
    corresponding points in a robust manner using RANSAC. After RANSAC, all the
    inlier correspondences will be used to re-estimate the homography matrix.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        thresh (float): Maximum allowed squared bidirectional pixel reprojection
          error to treat a point pair as an inlier (default: 16.0). Pixel
          reprojection error is computed as:
            d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
          where ||a|| denotes the l2 norm (euclidean distance) of vector a.
        num_tries (int): Number of trials for RANSAC

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.
        mask (np.ndarraay): Output mask with dtype np.bool where 1 indicates
          inliers

    Prohibited functions:
        cv2.findHomography()
    """
    maxn=0
    for i in range(num_tries):
        select=np.random.choice(src.shape[0], 4, replace=False)
        x = src[select]
        y= dst[select]
        H=compute_homography(x,y)
        mask=(compute_homography_error(src,dst,H)<thresh)
        if(np.sum(mask)>maxn):ans,maxn=mask,np.sum(mask)
        #inliers=np.sum(mask)
        #w=inliers/
        #N=np.log(0.01)/np.log(1-+0.0001)
    h_matrix=compute_homography(src[ans],dst[ans])

    """ YOUR CODE STARTS HERE """
    
    """ YOUR CODE ENDS HERE """

    return h_matrix, ans


# Part 4
def concatenate_homographies(pairwise_h_matrices, ref):
    """Transforms pairwise relative transformations to absolute transformations.

    Args:
        pairwise_h_matrices (list): List of N-1 pairwise homographies, the i'th
          matrix maps points in the i'th image to the (i+1)'th image, e.g..
          x_1 = H[0] * x_0
        ref (int): Reference image to warp all images towards.

    Returns:
        abs_h_matrices (list): List of N homographies. abs_H[i] warps points
           in the i'th image to the reference image. abs_H[ref] should be the
           identity transformation.
    """

    abs_h_matrices = []
    num_images = len(pairwise_h_matrices) + 1
    assert ref < num_images
    for i in range(len(pairwise_h_matrices)+1):
        #invH_list.append(np.linalg.inv(pairwise_h_matrices[i]))
        H=np.eye(pairwise_h_matrices[0].shape[0]).astype(np.float64)
        if(i<ref):
            for j in range(i,ref):
                H=pairwise_h_matrices[j].dot(H)
        elif(i > ref):
            for j in range(ref,i):
                H = H.dot(np.linalg.inv(pairwise_h_matrices[j]))
        abs_h_matrices.append(H)

    """ YOUR CODE STARTS HERE """
    
    """ YOUR CODE ENDS HERE """

    return abs_h_matrices

