import scipy.optimize as optimize
import numpy as np
from Utils.MiscUtils import *


def NonLinearPnP(K, p, x3D, R0, C0):
    Q = getQuaternion(R0)
    start_X = [Q[0], Q[1], Q[2], Q[3], C0[0], C0[1], C0[2]]

    optimizer = optimize.least_squares(fun=PnPErr, x0=start_X, method="trf", args=[x3D, p, K])
    X1 = optimizer.x
    Q = X1[:4]
    C = X1[4:]
    R = getRotation(Q)
    return R, C

def PnPErr(X0, x3D, p, K):
    Q, C = X0[:4], X0[4:].reshape(-1, 1)
    R = getRotation(Q)
    P = ProjectionMatrix(R, C, K)

    Err = []
    for X, pt in zip(x3D, p):
        p_1T, p_2T, p_3T = P
        p_1T, p_2T, p_3T = p_1T.reshape(1, -1), p_2T.reshape(1, -1), p_3T.reshape(1, -1)

        X = homo(X.reshape(1, -1)).reshape(-1, 1)
        u, v = pt[0], pt[1]
        u_proj = p_1T.dot(X) / p_3T.dot(X)
        v_proj = p_2T.dot(X) / p_3T.dot(X)

        E = (v - v_proj) ** 2 + (u - u_proj) ** 2

        Err.append(E)

    return np.mean(np.array(Err).squeeze())
