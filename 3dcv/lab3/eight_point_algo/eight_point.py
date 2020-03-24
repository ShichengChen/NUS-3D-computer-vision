'''
Name: Shicheng Chen
Email: e0534721@u.nus.edu
Student ID: A0215003A

'''



import numpy as np
import scipy.io as sio
import h5py
import cv2
import matplotlib.pyplot as plt


"""Helper functions: You should not have to touch the following functions.
"""
def compute_right_epipole(F):

    U, S, V = np.linalg.svd(F.T)
    e = V[-1]
    return e / e[2]


def plot_epipolar_line(img1, img2, F, x1, x2, epipole=None, show_epipole=False):
    """
    Visualize epipolar lines in the imame

    Args:
        img1, img2: two images from different views
        F: fundamental matrix
        x1, x2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate
    Returns:

    """
    plt.figure()
    plt.imshow(img1)
    for i in range(x1.shape[1]):
      plt.plot(x1[0, i], x1[1, i], 'bo')
      m, n = img1.shape[:2]
      line1 = np.dot(F.T, x2[:, i])
      t = np.linspace(0, n, 100)
      lt1 = np.array([(line1[2] + line1[0] * tt) / (-line1[1]) for tt in t])
      ndx = (lt1 >= 0) & (lt1 < m)
      plt.plot(t[ndx], lt1[ndx], linewidth=2)
    plt.figure()
    plt.imshow(img2)

    for i in range(x2.shape[1]):
      plt.plot(x2[0, i], x2[1, i], 'ro')
      if show_epipole:
        if epipole is None:
          epipole = compute_right_epipole(F)
        plt.plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


      m, n = img2.shape[:2]
      line2 = np.dot(F, x1[:, i])

      t = np.linspace(0, n, 100)
      lt2 = np.array([(line2[2] + line2[0] * tt) / (-line2[1]) for tt in t])

      ndx = (lt2 >= 0) & (lt2 < m)
      plt.plot(t[ndx], lt2[ndx], linewidth=2)
    plt.show()


def compute_essential(data1, data2, K):
    """
    Compute the essential matrix from point correspondences and intrinsic matrix

    Args:
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate
        K: intrinsic matrix of the camera
    Returns:
        E: Essential matrix
    """

    k_=np.linalg.inv(K)
    src=(k_ @ data1).T
    dst=(k_ @ data2).T
    src[:,:2]=src[:,:2]/src[:,2:3]
    dst[:,:2]=dst[:,:2]/dst[:,2:3]
    x, y = dst[:, 0:1], dst[:, 1:2]
    u, v = src[:, 0:1], src[:, 1:2]
    arr = np.concatenate([x * u, x * v, x, y * u, y * v, y, u, v, np.ones((src.shape[0], 1))], axis=1)
    U, S, Vh = np.linalg.svd(arr)
    E = Vh[-1, :].reshape(3, 3)
    U, S, Vh = np.linalg.svd(E)
    ss=S.copy()
    S[0],S[1],S[2]=(ss[0]+ss[1])/2,(ss[0]+ss[1])/2,0
    E=U@np.diag(S)@Vh
    """YOUR CODE STARTS HERE"""
    """YOUR CODE ENDS HERE"""
    return E


def decompose_e(E, K, data1, data2):
    """
    Compute the essential matrix from point correspondences and intrinsic matrix

    Args:
        E: Essential matrix
        K: intrinsic matrix of the camera
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate

    Returns:
        trans: 3x4 array representing the transformation matrix
    """
    x1,x2=data1[:,:],data2[:,:]
    U, S, Vh = np.linalg.svd(E)
    if(np.linalg.det(U)<0):U=-U
    if(np.linalg.det(Vh)<0):Vh=-Vh
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    r = U@W@Vh
    t=(U[:,-1]).reshape(U.shape[0],1)
    p2=K@np.concatenate([r,t], axis =1)
    p1=K@np.concatenate([np.eye(3),np.zeros_like(t)], axis =1)
    X1=np.linalg.pinv(p1)@x1
    X2=np.linalg.pinv(p2)@x2
    if(X1[2]<0).any() or (X2[2]<0).any():r=U@W.T@Vh
    p1 = K @ np.concatenate([np.eye(3), np.zeros_like(t)], axis=1)
    p2 = K @ np.concatenate([r, t], axis=1)
    X1 = np.linalg.pinv(p1) @ x1
    X2 = np.linalg.pinv(p2) @ x2
    if(X1[2]<0).any() or (X2[2]<0).any():t=-t
    #print(r,t)
    """YOUR CODE STARTS HERE"""
    #_, r, t, _ = cv2.recoverPose(E, data1[:2, :].T, data2[:2, :].T, K)
    trans = np.concatenate([r, t], axis =1)
    """YOUR CODE ENDS HERE"""
    #print(r, t)
    return trans


def compute_fundamental(data1, data2):
    """
    Compute the fundamental matrix from point correspondences

    Args:
        data1, data2: 3x15 arrays containing 15 point correspondences in homogeneous coordinate

    Returns:
        F: fundamental matrix
    """
    def transform_homography(src, h_matrix):
        home = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
        transformed = (h_matrix @ home.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        return transformed
    src,dst=data1.copy(),data2.copy()
    src = (src[:2, :]/src[2, :]).T
    dst = (dst[:2, :]/dst[2, :]).T
    N = src.shape[0]
    c0 = np.array([[src[:, 0].sum() / N, src[:, 1].sum() / N]])
    c1 = np.array([[dst[:, 0].sum() / N, dst[:, 1].sum() / N]])
    s0 = np.sqrt(2) / np.mean(np.sqrt(np.sum((src - c0) ** 2,axis=1)))
    s1 = np.sqrt(2) / np.mean(np.sqrt(np.sum((dst - c1) ** 2,axis=1)))
    N0 = np.array([[s0, 0, -s0 * c0[0, 0]], [0, s0, -s0 * c0[0, 1]], [0, 0, 1]])
    N1 = np.array([[s1, 0, -s1 * c1[0, 0]], [0, s1, -s1 * c1[0, 1]], [0, 0, 1]])
    osrc,odst=src.copy(),dst.copy()
    src = transform_homography(src, N0)
    dst = transform_homography(dst, N1)
    num_tries=30
    maxn=1,
    def get_f(se):
        x, y = dst[se][:, 0:1], dst[se][:, 1:2]
        u, v = src[se][:, 0:1], src[se][:, 1:2]
        arr = np.concatenate([x * u, x * v, x, y * u, y * v, y, u, v, np.ones((se.shape[0], 1))], axis=1)
        U, S, Vh = np.linalg.svd(arr)
        f = Vh[-1, :].reshape(3, 3)
        U, S, Vh = np.linalg.svd(f)
        S[-1] = 0
        f = U @ np.diag(S) @ Vh
        f = N1.T @ f @ N0
        f = f / f[-1, -1]
        return f
    for i in range(num_tries):
        se=np.random.choice(src.shape[0], 8, replace=False)
        f = get_f(se)
        xh = np.concatenate([osrc, np.ones((src.shape[0], 1))], axis=1)
        uh = np.concatenate([odst, np.ones((src.shape[0], 1))], axis=1)
        ans = uh @ f @ xh.T
        eps=1e-2
        mask=np.abs(np.diag(ans))<eps
        if(maxn<np.sum(mask)):
            maxn=np.sum(mask)
            valid=np.arange(src.shape[0])[mask]
    f=get_f(valid)
    """YOUR CODE STARTS HERE"""
    #F, _ = cv2.findFundamentalMat(data1[:2, :].T, data2[:2, :].T, method = cv2.FM_8POINT)
    """YOUR CODE ENDS HERE"""
    return f











