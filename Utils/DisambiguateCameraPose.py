import numpy as np
from Utils.LinearTriangulation import *

#checks for positive depth and chirality 
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


def depthPositivityConstraint(pts , r3 , C):
    n = 0
    for X in pts:
        X = X.reshape(-1,1)
        if np.dot(r3 , (X-C)) > 0  and X[2]>0:  
            n += 1
    return n

def recoverPose(E , points1 , points2 , k1 , k2):
    #get3d points using linear triangulation method
    Rset , Cset = decomposeE(E)
    points3d = linearTriangulation(Rset , Cset , points1 , points2 , k1,k2)
    best = 0
    max_depth = 0
    for i in range(len(Rset)):
        R , C = Rset[i] , Cset[i].reshape(-1,1)
        r3 = R[2].reshape(1,-1)
        pt3d = points3d[i]
        n = depthPositivityConstraint(pt3d , r3 , C)
        if n >  max_depth:
            best = i
            max_depth = n
        
    R , C , pts3d = Rset[best] , Cset[best] , points3d[best]
    return R , C , pts3d


def disambiguatePose(r_set, c_set, x3D_set):
    best_i = 0
    max_positive_depths = 0
    
    for i in range(len(r_set)):
        R, C = r_set[i],  c_set[i].reshape(-1,1) 
        r3 = R[2, :].reshape(1,-1)
        x3D = x3D_set[i]
        x3D = x3D / x3D[:,3].reshape(-1,1)
        x3D = x3D[:, 0:3]
        n_positive_depths = depthPositivityConstraint(x3D, r3,C)
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths
#         print(n_positive_depths, i, best_i)

    R, C, x3D = r_set[best_i], c_set[best_i], x3D_set[best_i]

    return R, C, x3D 