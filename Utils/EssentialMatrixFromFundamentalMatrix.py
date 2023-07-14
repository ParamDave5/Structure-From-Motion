import cv2
import numpy as np
K = [[568.996140852 , 0 , 643.21055941 ] ,
    [0 , 568.99362396 , 477.982801038] , 
    [0 , 0 , 1]]

K = np.array(K)

def essentialMatrix(F , K):
    return np.dot(K.T , np.dot(F , K))

def getEssentialMatrix(K , F):
    E = K.T.dot(F).dot(K)
    u , s , v = np.linalg.svd(E)
    s = [1,1,0]
    E_final = np.dot(u , np.dot(np.diag(s),v))
    return E_final



