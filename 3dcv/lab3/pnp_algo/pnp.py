import numpy as np
import cv2
from sympy.polys import subresultants_qq_zz


"""Helper functions: You should not have to touch the following functions.
"""
def extract_coeff(x1, x2, x3, cos_theta12, cos_theta23, cos_theta13, d12, d23, d13):
    """
    Extract coefficients of a polynomial

    Args:
        x1, x2, x3: symbols representing the unknown camera-object distance
        cos_theta12, cos_theta23, cos_theta13: cos values of the inter-point angles
        d12, d23, d13: square of inter-point distances

    Returns:
        a: the coefficients of the polynomial of x1
    """
    f12 = x1 ** 2 + x2 ** 2 - 2 * x1 * x2 * cos_theta12 - d12
    f23 = x2 ** 2 + x3 ** 2 - 2 * x2 * x3 * cos_theta23 - d23
    f13 = x1 ** 2 + x3 ** 2 - 2 * x1 * x3 * cos_theta13 - d13
    matrix = subresultants_qq_zz.sylvester(f23, f13, x3)
    f12_ = matrix.det()
    f1 = subresultants_qq_zz.sylvester(f12, f12_, x2).det()
    a1 = f1.func(*[term for term in f1.args if not term.free_symbols])
    a2 = f1.coeff(x1 ** 2)
    a3 = f1.coeff(x1 ** 4)
    a4 = f1.coeff(x1 ** 6)
    a5 = f1.coeff(x1 ** 8)
    a = np.array([a1, a2, a3, a4, a5])
    return a



def icp(points_s, points_t):
    """
    Estimate the rotation and translation using icp algorithm

    Args:
        points_s : 10 x 3 array containing 3d points in the world coordinate
        points_t : 10 x 3 array containing 3d points in the camera coordinate

    Returns:
        r: rotation matrix of the camera
        t: translation of the camera
    """
    us = np.mean(points_s, axis=0, keepdims=True)
    ut = np.mean(points_t, axis=0, keepdims=True)
    points_s_center = points_s - us
    points_t_center = points_t - ut
    w = np.dot(points_s_center.T, points_t_center)
    u, s, vt = np.linalg.svd(w)
    r = vt.T.dot(u.T)
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T.dot(u.T)
    t = ut.T - np.dot(r, us.T)
    return r, t

def reconstruct_3d(X, K, points2d):
    """
    Reconstruct the 3d points from camera-point distance

    Args:
        X: a list containing camera-object distances for all points
        K: intrinsics of camera
        points2d: 10x1x3 array containing 2d coordinates of points in the homogeneous coordinate

    Returns:
        points3d_c: 3d coordinates of all points in the camera coordinate
    """
    points3d_c = []
    for i in range(len(X)):
        points3d_c.append(X[i] * np.dot(np.linalg.inv(K), points2d[i].T))
    points3d_c = np.hstack(points3d_c)
    return points3d_c

def visualize(r, t, points3d, points2d, K):
    """
    Visualize reprojections of all 3d points in the image and compare with ground truth

    Args:
        r: rotation matrix of the camera
        t: tranlation of the camera
        points3d:  10x3 array containing 3d coordinates of points in the world coordinate
        points3d:  10x2 array containing ground truth 2d coordinates of points in the image space
    """
    scale = 0.2
    img = cv2.imread('data/img_id4_ud.JPG')
    dim = (int(img.shape[1]*scale), int(img.shape[0]*scale))
    img = cv2.resize(img, dim)
    trans = np.hstack([r, t])
    points3d_homo = np.hstack([points3d, np.ones((points3d.shape[0], 1))])
    points2d_re = np.dot(K, np.dot(trans, points3d_homo.T))
    points2d_re = np.transpose(points2d_re[:2, :]/points2d_re[2:3, :])
    for j in range(points2d.shape[0]):
        cv2.circle(img, (int(points2d[j, 0]*scale), int(points2d[j, 1]*scale)), 3,  (0, 0, 255))
        cv2.circle(img, (int(points2d_re[j, 0]*scale), int(points2d_re[j, 1]*scale)), 4,  (255, 0, 0))
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pnp_algo(K, points2d, points3d):
    """
    Estimate the rotation and translation of camera by using pnp algorithm

    Args:
        K: intrinsics of camera
        points2d: 10x1x2 array containing 2d coordinates of points in the image space
        points3d: 10x1x3 array containing 3d coordinates of points in the world coordinate
    Returns:
        r: 3x3 array representing rotation matrix of the camera
        t: 3x1 array representing translation of the camera
    """

    # def perspective_projection(xyz_point, camera):
    #     if xyz_point.ndim == 1:
    #         uvd_point = np.zeros((3))
    #         uvd_point[0] = xyz_point[0] * camera.fx / xyz_point[2] + camera.cx
    #         uvd_point[1] = xyz_point[1] * camera.fy / xyz_point[2] + camera.cy
    #         uvd_point[2] = xyz_point[2]
    print("k",K)


    import sympy as sym
    import itertools
    n = 5
    n=points2d.shape[0]
    points2d=points2d[:n]
    points3d=points3d[:n]
    f,cx,cy=K[0,0],K[0,-1],K[1,-1]
    dis = np.zeros((n,))
    #for se in itertools.combinations(np.arange(1,points2d.shape[0]), 2):
    for start in range(n):
        rest=np.arange(n).tolist()
        rest.remove(start)
        arr = []
        for se in itertools.combinations(rest, 2):
            x1, x2, x3 = sym.symbols('x1, x2, x3')
            se=list(se)
            se=[start]+se
            print(se)
            d12=np.sum((points3d[se[0],:,:]-points3d[se[1],:,:])**2)
            d23=np.sum((points3d[se[1],:,:]-points3d[se[2],:,:])**2)
            d13=np.sum((points3d[se[0],:,:]-points3d[se[2],:,:])**2)

            u1, v1 = (points2d[se[0], 0, 0]-cx)/f, (points2d[se[0], 0, 1]-cy)/f
            norm = 1/np.sqrt(u1 * u1 + v1 * v1 + 1)
            j1=np.array([u1*norm,v1*norm,norm])

            u2, v2 = (points2d[se[1], 0, 0]-cx) /f, (points2d[se[1], 0, 1]-cy) / f
            norm = 1 / np.sqrt(u2 * u2 + v2 * v2 + 1)
            j2 = np.array([u2 * norm, v2 * norm, norm])
            u3, v3 = (points2d[se[2], 0, 0] - cx) / f, (points2d[se[2], 0, 1]- cy) / f
            norm = 1 / np.sqrt(u3 * u3 + v3 * v3 + 1)
            j3 = np.array([u3 * norm, v3 * norm, norm])

            cos_theta12, cos_theta23, cos_theta13 = np.sum(j1*j2), np.sum(j2*j3), np.sum(j1*j3)
            a=extract_coeff(x1,x2,x3,cos_theta12,cos_theta23,cos_theta13,d12,d23,d13)
            arr.append(a)
        arr=np.array(arr).astype(np.float)
        U, S, Vh = np.linalg.svd(arr)
        t5 = Vh[-1, :]
        #print(t5)
        #print([t5[1]/t5[0],t5[2]/t5[1],t5[3]/t5[2],t5[4]/t5[3]])
        dis[start]=np.sqrt(np.mean([t5[1]/t5[0],t5[2]/t5[1],t5[3]/t5[2],t5[4]/t5[3]]))


    points3dcam=reconstruct_3d(dis,K,np.concatenate([points2d,np.ones([n,1,1])],axis=2))
    points3dcam=points3dcam.T
    r,t=icp(np.squeeze(points3d),np.squeeze(points3dcam))
    print(r, t)
    """YOUR CODE STARTS HERE"""
    # _, r, t = cv2.solvePnP(points3d, points2d, K, np.zeros((5,)))
    # r=cv2.Rodrigues(r)[0]
    # print(cv2.Rodrigues(r)[0].shape)
    # print(cv2.Rodrigues(r)[1].shape)
    # """YOUR CODE ENDS HERE"""
    # print(r,t)
    return r, t













