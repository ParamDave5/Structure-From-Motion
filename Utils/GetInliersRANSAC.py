import numpy as np
import cv2 
from Utils.EstimateFundamentalMatrix import *

def epipolarError(pt1 , pt2 , F):
    pt1 = np.array([pt1[0] , pt1[1] , 1])
    pt2 = np.array([pt2[0] , pt2[1] , 1]).T
    error = np.dot(pt2 , np.dot(F,pt1))

    return np.abs(error)

def ransacInliers(points1 , points2 ,idx ):
    n_iter = 2000
    error_thresh = 0.05
    max_inliers = 0
    choosen_indices = []
    choosen_f = 0
    n_rows = points1.shape[0]

    for i in range(n_iter):
        indices = []
        random_indices = np.random.choice(n_rows , size = 8)
        points1_8 = points1[random_indices , : ]
        points2_8 = points2[random_indices , : ]

        F = fundamentalMatrix(points1_8 , points2_8)
        # print(F)
        if F is not None:
            for j in range(n_rows):
                error = epipolarError(points1[j , :] , points2[j , :] , F)
                # print("doing error " ,error)
                if error < error_thresh:
                    indices.append(idx[j])

        if len(indices) > max_inliers:
            max_inliers = len(indices)
            choosen_indices = indices
            F_final = F
   
    # pts1_inliers , pts2_inliers = points1[inliers] , points2[inliers]
    return F_final , choosen_indices

def ransac(points1 , points2 ):
    n_iter = 1000
    error_thresh = 0.05
    # points1norm , T1 = normalize(points1)
    # points2norm , T2 = normalize(points2)

    max_inliers = 0
    n_rows = points1.shape[0]
    inliers = []
    for i in range(n_iter):
        indices = []
        random_indices = np.random.choice(n_rows , size = 8)
        points1_8 = points1[random_indices]
        points2_8 = points2[random_indices]

        F = fundamentalMatrix(points1_8 , points2_8)
        # print(F)
        for j in range(n_rows):
            error = epipolarError(points1[j] , points2[j] , F)
            if abs(error) < error_thresh:
                indices.append(j)
        # print(len(indices))
        if len(indices) > max_inliers:
            max_inliers = len(indices)
            inliers = indices
            F_final = F
    # print(len(inliers))
    # F_ = getF(F_final , T2 , T1)
    pts1_inliers , pts2_inliers = points1[inliers] , points2[inliers]
    return F_final , pts1_inliers , pts2_inliers
