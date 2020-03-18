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
    print(data2.shape)
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

    print(np.abs(np.diag(dst @ E @ src.T)))
    print(E)
    """YOUR CODE STARTS HERE"""
    E, _ = cv2.cv2.findEssentialMat(data1[:2, :].T, data2[:2, :].T, cameraMatrix=K)
    """YOUR CODE ENDS HERE"""
    print(np.abs(np.diag(src @ E @ dst.T)))
    print(E)
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
    U, S, Vh = np.linalg.svd(E)
    Z=np.array([[0,1,0],[-1,0,0],[0,0,0]])
    W=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    r=U@W.T@Vh
    t=(-U[:,-1]).reshape(U.shape[0],1)
    #print(r,t)
    """YOUR CODE STARTS HERE"""
    #_, r, t, _ = cv2.recoverPose(E, data1[:2, :].T, data2[:2, :].T, K)
    # print(r.shape)
    # print(t.shape)
    trans = np.concatenate([r, t], axis =1)
    """YOUR CODE ENDS HERE"""
    print(trans)
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
    print('N',N)
    c0 = np.array([[src[:, 0].sum() / N, src[:, 1].sum() / N]])
    c1 = np.array([[dst[:, 0].sum() / N, dst[:, 1].sum() / N]])
    s0 = np.sqrt(2) / np.sum(np.sqrt((src - c0) ** 2).sum() / N)
    s1 = np.sqrt(2) / np.sum(np.sqrt((dst - c1) ** 2).sum() / N)
    N0 = np.array([[s0, 0, -s0 * c0[0, 0]], [0, s0, -s0 * c0[0, 1]], [0, 0, 1]])
    N1 = np.array([[s1, 0, -s1 * c1[0, 0]], [0, s1, -s1 * c1[0, 1]], [0, 0, 1]])
    osrc,odst=src.copy(),dst.copy()
    src = transform_homography(src, N0)
    dst = transform_homography(dst, N1)
    #print(dst.shape)
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
        #print(np.diag(ans))
        mask=np.abs(np.diag(ans))<eps
        if(maxn<np.sum(mask)):
            maxn=np.sum(mask)
            valid=np.arange(src.shape[0])[mask]
    print('valid',valid)
    f=get_f(valid)
    print('f',f)
    #return f

    # [[3.05524487e-06 -5.51310423e-06  3.00972423e-03]
    #  [8.46939076e-06  3.59803044e-06 -3.50950924e-04]
    # [-5.22853679e-03 -3.92238487e-03 1.00000000e+00]]

    # [[2.87100835e-06 - 4.57449434e-06  3.61903377e-03]
    #  [9.13336953e-06  2.04724332e-06  2.79020676e-03]
    # [-5.95286860e-03 - 6.69818358e-03
    # 1.00000000e+00]]

    """YOUR CODE STARTS HERE"""

    F, _ = cv2.findFundamentalMat(data1[:2, :].T, data2[:2, :].T, method = cv2.FM_8POINT)
    """YOUR CODE ENDS HERE"""
    print('F',f)
    return F











