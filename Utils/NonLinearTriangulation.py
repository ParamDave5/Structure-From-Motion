import numpy as np
import scipy.optimize as optimize

from Utils.BundleAdjustment import projection

def NonLinearTriangulation(K, p1_list, p2_list, initial_x3D, R1, C1, R2, C2):
    #convert matrices into a single projection matrix
    P1 = Projection(R1, C1, K)
    P2 = Projection(R2, C2, K)
    if p1_list.shape[0] != p2_list.shape[0] != initial_x3D.shape[0]:
        raise ' Chech point dimensions - level nlt'
    refined_x3D = []

    for i in range(len(initial_x3D)):
        optima = optimize.least_squares(fun=reprojectionError, x0=initial_x3D[i], method="trf",
                                        args=[p1_list[i], p2_list[i], P1, P2])
        
        refined_x3D.append(optima.x)
    return np.array(refined_x3D)

def meanReprojectionError(x3D, pts1, pts2, R1, C1, R2, C2, K ):  
    
    # print("print shape of x3d " , x3D)
    Error = []
    P1 = Projection(R1, C1, K)
    P2 = Projection(R2, C2, K)
    pts1 = np.array(pts1 , dtype = np.int32)
    pts2 = np.array(pts2 , dtype = np.int32)
    x3D = np.array(x3D , dtype = np.int32)

    for pt1, pt2, X in zip(pts1, pts2, x3D):
        # print(pt1 , pt2 , X)
        e1 = reprojectionError(X, pt1, pt2, P1 , P2 )
        # print("errors " ,e1)
        Error.append(e1)
    Error = np.array(Error)
    col_mean = np.nanmean(Error, axis = 0)
    inds = np.where(np.isnan(Error))
    Error[inds] = col_mean
    return np.mean(Error)

#reprojecting the points. Calculating the error (u_i - Pv_i)**2 
def reprojectionError(X, p1, p2, P1 , P2):

    p1_1T, p1_2T, p1_3T = P1[0].reshape(1, -1), P1[1].reshape(1, -1), P1[2].reshape(1, -1)
    p2_1T, p2_2T, p2_3T = P2[0].reshape(1, -1), P2[1].reshape(1, -1), P2[2].reshape(1, -1)
    u1, v1 = p1[0], p1[1]
   
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
    v1_proj = np.divide(p1_2T.dot(X) , p1_3T.dot(X))
    
    # E1 = (v1 - v1_proj) ** 2 + (u1 - u1_proj) ** 2
    E1 = np.square(v1 - v1_proj) + np.square(u1 - u1_proj)
    # print(int(E1))
    if E1 is None:
        E1 = 0

    u2, v2 = p2[0], p2[1]
    # u2_proj = p2_1T.dot(X) / p2_3T.dot(X)
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    # v2_proj = p2_2T.dot(X) / p2_3T.dot(X)
    v2_proj = np.divide(p2_2T.dot(X) , p2_3T.dot(X))
    # E2 = (v2 - v2_proj) ** 2 + (u2 - u2_proj) ** 2

    E2 = np.square(v2 - v2_proj) + np.square(u2 - u2_proj)

    # return np.square(E1 + E2)
    
    error = E1 + E2 
    return error.squeeze()

# K.R.[I | -C]
def Projection(R, C, K):
    C = C.reshape(3, 1)
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P
