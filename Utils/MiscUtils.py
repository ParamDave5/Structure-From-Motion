import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import os
import matplotlib.pyplot as plt

def ProjectionMatrix(R, C, K):
    C = np.reshape(C, (3, 1))
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def getQuaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def getEuler(R):
    Q = Rotation.from_matrix(R)
    return Q.as_quat()

def getRotation(Q, type_='q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def giveXYZ(pts):
    x = []
    y = []
    z = []
    for i in pts:
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    return x , y , z

def split(pts3d):
    X = []
    Y =[]
    Z = []
    for i in range(pts3d.shape[0]):
        for j in range(pts3d.shape[1]):
            X.append(pts3d[i][j][0])
            Y.append(pts3d[i][j][1])
            Z.append(pts3d[i][j][2])

    return X , Y , Z


    