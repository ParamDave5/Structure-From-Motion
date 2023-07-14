import numpy as np


#return lists of all R and C
def decomposeE(E):
    u,s,vt = np.linalg.svd(E)
    W = np.array([[0,-1,0] , [1,0,0] , [0,0,1]])
    c1 = u[:,2]
    r1 = np.dot(u , np.dot(W , vt))
    c2 = -u[:,2]
    r2 = np.dot(u , np.dot(W , vt))
    c3 = u[:,2]
    r3 = np.dot(u , np.dot(W.T , vt))
    c4 = -u[:,2]
    r4 = np.dot(u , np.dot(W.T , vt))

    R = np.array([r1,r2,r3,r4] , dtype =np.float32)
    C = np.array([c1,c2,c3,c4] , dtype = np.float32)
    for i in range(4):
        if np.linalg.det(R[i]) < 0:
            R[i] = -R[i]
            C[i] = -C[i]
    return R , C


