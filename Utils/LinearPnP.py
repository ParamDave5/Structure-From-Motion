import numpy as np

def ProjectionMatrix(R , K , C):
    C = np.reshape(C , (3,1))
    I = np.identity(3)
    P = np.dot(K , np.dot(R , np.hstack((I , -C))))
    return P

#given X , x and K ;we find the C and R for a given set of points
def another_Homo(x):
    c, w = x.shape
    if w == 3 or w == 2:
        o = np.ones((c, 1))
        x_new = np.concatenate((x, o), axis=1)
    else:
        x_new = x
    return x_new

def reprojErrorPnP(x3D , pts , K , R , C):
    P = ProjectionMatrix(R ,K , C)
    error = []

    for X , pt in zip(x3D , pts):
        pt1 , pt2 , pt3 = P
        pt1 ,pt2 , pt3 = pt1.reshape(1,-1) , pt2.reshape(1,-1) , pt3.reshape(1,-1)
        # print("point pt1: " , pt1.shape)
        # print("point pt2: " , pt2.shape)
        # print("point pt3: " , pt3.shape)


        X = another_Homo(X.reshape(1,-1)).reshape(-1,1)
        u , v = pt[0] , pt[1]
        # print("shape pf pt1.X",pt1.dot(X).shape)
        # print("shape pf pt3.X",pt3.dot(X).shape)

        u_proj = np.divide(pt1.dot(X) , pt3.dot(X))
        v_proj = np.divide(pt2.dot(X) , pt3.dot(X))
        E = np.square(v - v_proj) + np.square(u - u_proj)

        error.append(E)
    mean_error = np.mean(np.array(error).squeeze())
    return mean_error


def combine_intrinsic_extrinsic(x, K, C, R):
    C = C.reshape(-1, 1)
    x = x.reshape(-1, 1)
    P = K @ R @ np.concatenate((np.identity(3), -C), axis=1)
    X = np.vstack((x, 1))
    u = (P[0, :] @ X).T / (P[2, :] @ X).T
    v = (P[1, :] @ X).T / (P[2, :] @ X).T
    return np.concatenate((u, v), axis=0)

#final code output
def LinearPnP(X, x, K):
    I = np.ones((X.shape[0], 1))  # 4x1
    X = np.concatenate((X, I), axis=1) # XI along column

    i = np.ones((x.shape[0] ,1 ))
    x = np.concatenate((x, i), axis=1) # xi along column

    x = np.transpose(np.linalg.inv(K) @ x.T)
    M = []
    for i in range(X.shape[0]):
        row_1 = np.hstack(
            (np.hstack((np.zeros((1, 4)), -X[i, :].reshape((1, 4)))), x[i, :][1] * X[i, :].reshape((1, 4))))
        row_2 = np.hstack(
            (np.hstack((X[i, :].reshape((1, 4)), np.zeros((1, 4)))), -x[i, :][0] * X[i, :].reshape((1, 4))))
        row_3 = np.hstack((np.hstack((-x[i, :][1] * X[i, :].reshape((1, 4)),
                                      x[i, :][0] * X[i, :].reshape((1, 4)))), np.zeros((1, 4))))
        A = np.vstack((np.vstack((row_1, row_2)), row_3))
        if (i == 0):
            M = A
        else:
            M = np.concatenate((M, A), axis=0)

    U, S, V_T = np.linalg.svd(M)
    R = V_T[-1].reshape((3, 4))[:, 0: 3]
    u, s, v_T = np.linalg.svd(R)
    np.identity(3)[2][2] = np.linalg.det(np.matmul(u, v_T))
    R = np.dot(np.dot(u, np.identity(3)), v_T)
    C = -np.dot(np.linalg.inv(R), V_T[-1].reshape((3, 4))[:, 3])
    if np.linalg.det(R) < 0:
        R = -R
        C = -C
    return C, R
