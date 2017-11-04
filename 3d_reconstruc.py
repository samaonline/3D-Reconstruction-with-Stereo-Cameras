import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import sys

def fundamental_matrix(matches):
    N = len(matches)
    x1 = matches[:, 0:2]
    x2 = matches[:, 2:4]
    ones = np.ones((N, 1))
    x1_homo = np.concatenate((x1, ones), axis=1)
    x2_homo = np.concatenate((x2, ones), axis=1)

    def construct_T(x):
        mu = np.mean(x, axis=0)
        std = np.std(np.linalg.norm(x, axis=1))
        T = np.eye(3)
        T[:2, :2] = np.eye(2) / std
        T[:2, 2] = -mu / std
        return T

    T1, T2 = map(construct_T, (x1, x2))

    x1_norm = np.dot(T1, x1_homo.T).T
    x2_norm = np.dot(T2, x2_homo.T).T
    
    # least square to calculate F_star
    A = np.array([
        np.outer(x2i, x1i).reshape((9,)) for x1i, x2i in zip(x1_norm, x2_norm)
    ])
    _, _, V_T = np.linalg.svd(A)
    F_star = V_T[-1].reshape((3, 3))
    
    # reduce rank to 2
    U, s, V_T = np.linalg.svd(F_star)
    s[-1] = 0
    F = np.dot(U, np.diag(s)).dot(V_T)

    # add normalization
    F = T2.T.dot(F).dot(T1)
    
    # calculate residual
    d12 = np.sum(x1_homo * x2_homo.dot(F.T), axis=1) \
        / np.linalg.norm(x2_homo.dot(F.T), axis=1)
    d21 = np.sum(x2_homo * x1_homo.dot(F.T), axis=1) \
        / np.linalg.norm(x1_homo.dot(F.T), axis=1)
    residual = np.mean(d12 ** 2 + d21 ** 2) / 2
    return F, residual

def find_rotation_translation(E):
    U, s, V_T = np.linalg.svd(E)
    R_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Rs = []
    ts = []
    t = U[:, -1]
    for U_signed in U, -U:
        for neg_indices in [
            (1, 1, 1), 
            (-1, 1, 1),
            (1, 1, -1),
            (1, -1, -1)
        ]:
            U_i = U_signed * np.array(neg_indices).reshape((1, 3))
            V_i_T = V_T * np.array(neg_indices).reshape((3, 1))
            R = U_i.dot(R_90.T.dot(V_i_T))
            if np.linalg.det(R) < 0:
                continue
            ts.append(U_i[:, -1])
            Rs.append(R)
    return ts, Rs

def find_3d_points(P1, P2, matches):
    points1 = matches[:, : 2]
    points2 = matches[:, 2 : 4]
    def reconstruction_error(points_3d, points_2d, P):
        ones = np.ones((len(points_3d), 1))
        points_3d_homo = np.concatenate((points_3d, ones), axis=1)
        pred_2d = np.dot(P, points_3d_homo.T).T
        pred_2d = pred_2d[:, :2] / pred_2d[:, 2 : 3]
        return np.mean(np.linalg.norm(points_2d - pred_2d, axis=1))
    td_points = []
    
    for (x1, y1), (x2, y2) in zip(points1, points2):
        P = np.zeros((4, 4))
        P[0] = P1[0] - x1 * P1[2]
        P[1] = P1[1] - y1 * P1[2]
        P[2] = P2[0] - x2 * P2[2]
        P[3] = P2[1] - y2 * P2[2]
        _, _, V_T = np.linalg.svd(P)
        point = V_T[-1]
        point = point[0:3] / point[3]
        td_points.append(point)
    td_points = np.array(td_points)
    rec_err1 = reconstruction_error(td_points, points1, P1)
    rec_err2 = reconstruction_error(td_points, points2, P2)
    return td_points, (rec_err1 + rec_err2) / 2

def plot_3d(points, camera1, camera2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points.T, c='blue')
    ax.scatter(*camera1, c='red')
    ax.scatter(*camera2, c='green')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    min_x, max_x = min(points[:, 0]), max(points[:, 0])
    min_y, max_y = min(points[:, 1]), max(points[:, 1])
    min_z, max_z = min(points[:, 2]), max(points[:, 2])
    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z)
    avg_x = (min_x + max_x) / 2
    avg_y = (min_y + max_y) / 2
    avg_z = (min_z + max_z) / 2
    ax.set_xlim([min(avg_x - max_range / 2, camera1[0], camera1[0]), 
                 max(avg_x + max_range / 2, camera1[0], camera1[0])])
    ax.set_ylim([min(avg_y - max_range / 2, camera1[1], camera1[1]),
                 max(avg_y + max_range / 2, camera1[1], camera1[1])])
    ax.set_zlim([min(avg_z - max_range / 2, camera1[2], camera1[2]),
                 max(avg_z + max_range / 2, camera1[2], camera1[2])])
    plt.show()
    
def reconstruct_3d(object):
    dir = './data/%s/' % object
    matches = scipy.io.loadmat(dir + '%s_matches.mat' % object)['points'] #np.loadtxt(dir + '%s_matches.txt' % object)
    K1 = scipy.io.loadmat(dir + '%s1_K.mat' % object)['fid']
    K2 = scipy.io.loadmat(dir + '%s2_K.mat' % object)['fid']
    house1 = mpimg.imread(dir + '%s1.jpg' % object)
    house2 = mpimg.imread(dir + '%s2.jpg' % object)

    plt.imshow(np.hstack((house1, house2)))
    width = house1.shape[1]
    for x1, y1, x2, y2 in matches:
        plt.scatter([x1, x2 + width], [y1, y2])
    #     plt.plot([x1, x2 + width], [y1, y2], color='r')
    plt.axis('off')
    plt.show()
    
#     %matplotlib notebook

    F, residual = fundamental_matrix(matches)
    print('F residual:', residual)

    E = K2.T.dot(F).dot(K1)
    P1 = K1.dot(np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1))
    best_n = -float('inf')
    best_tR = None
    for i, (t, R) in enumerate(zip(*find_rotation_translation(E))):
        print('Set', i + 1, ':')
        print('R =', R)
        print('t =', t)
        print()
        P2 = K2.dot(np.concatenate((R, t[:, None]), axis=1))
        points1, err = find_3d_points(P1, P2, matches)
        points2 = np.dot(points1, R.T) + t
        z_all = np.concatenate((points1[:, 2:], points2[:, 2:]), axis=1)
        n_front = np.sum(np.all(z_all > 0, axis=1))
        if n_front > best_n:
            best_n = n_front
            best_tR = (t, R, points1, err)
    t, R, points, err = best_tR
    print('Reconstruction error:', err)
    
    plot_3d(points, np.zeros((3,)), -R.T.dot(t))

object_name = sys.argv[1]
reconstruct_3d(object_name)