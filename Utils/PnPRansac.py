import numpy as np
import random
from Utils.LinearPnP import LinearPnP, another_Homo, combine_intrinsic_extrinsic

def ProjectionMatrix(R , K , C):
    C = np.reshape(C , (3,1))
    I = np.identity(3)
    P = np.dot(K , np.dot(R , np.hstack((I , -C))))
    return P

def PnPerror(x, X , R ,C ,K ):
    u,v = x 
    X = another_Homo(X.reshape(-1,1)).reshape(-1,1)
    C = C.reshape(-1,1)
    P = ProjectionMatrix(R , K , C)
    p1 , p2 , p3 = P

    u_p = np.divide(p1.dot(X) , p3.dot(X))
    v_p = np.divide(p2.dot(X) , p3.dot(X))

    x_p = np.hstack((u_p,v_p))
    x = np.hstack((u,v))
    e = np.linalg.norm(x - x_p)
    return e


def PnPRANSAC(X, x, K):
    C_new = np.zeros((3, 1))
    R_new = np.identity(3)
    n = 0
    epsilon = 5
    x_H = another_Homo(x)
    M = 500
    N = x.shape[0]


    for _ in range(M):
        shape = min(x.shape[0] , X.shape[0])
        # print("minimum of the two X and x is: ", shape)
        random_idx = random.sample(range(shape), 6)
        
        # print("random indices X: ",X[random_idx])
        # print("random indices: ", x[random_idx])
        C, R = LinearPnP(X[random_idx], x[random_idx], K)
 
        C , R = np.array(C).reshape(-1,1) , np.array(R).reshape(3,3) 
        
        S = []
        for j in range(N):
            re_p = combine_intrinsic_extrinsic(x_H[j][:], K, C, R)
            e = np.sqrt(np.square((x_H[j, 0]) - re_p[0]) + np.square((x_H[j, 1] - re_p[1])))
            if e < epsilon:
                S.append(j)
        abs_S = len(S)

        if n < abs_S:
            n = abs_S
            R_new = R
            C_new = C

        if abs_S == x.shape[0]:
            break
    return C_new, R_new
