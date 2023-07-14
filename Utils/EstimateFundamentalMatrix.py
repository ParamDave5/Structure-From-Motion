import numpy as np
import cv2


def seperate(points):
    x = []
    y = []
    for i in points:
        x.append(i[0])
        y.append(i[1])
    return x,y

#estimate fundamental matrix using 8-point method for a set of points

def fundamentalMatrix(points1 , points2):
    #create a matrix
    
    points1 , T1 = normalize(points1)
    points2 , T2 = normalize(points2)
    A = []
    for p1,p2 in zip(points1 , points2):
        x1 , y1 = p1[0] , p1[1]
        x2 , y2 = p2[0] , p1[1]
        a = [x1*x2 , x1*y2 , x1 , y1*x2 , y1*y2 , y1 , x2 , y2 , 1]
        # a = [x1*x2 , x2*y1 , x2 , y2*x1 , y2*y1 , y2 , x1 , y1 , 1]
        A.append(a)

    U , sigma , vt = np.linalg.svd(A,full_matrices=True)
    F = vt[-1:]
    F  = F.reshape(3,3)

    U ,sigma , vt = np.linalg.svd(F)
    sigma = np.diag(sigma)
    sigma[2,2] = 0
    F_ = np.dot(U , np.dot(sigma , vt))
    F_final = np.dot(T2.T , np.dot(F,T1))
    F_final = F_final/F_final[2,2]
    return F_final

#calculate epipolar errror x_TFx = error for a set of points
def calculateError(points1 , points2 , F):
    error = []
    for point1 , point2 in zip(points1 , points2):
        point1 = np.array([point1[0] , point1[1] , 1])
        point2 = np.array([point2[0] , point2[1] , 1]).T
        # err = abs(np.squeeze( np.matmul( (np.matmul(point2,F) ,point1) ) ) )
        err = abs(np.dot(points2.T , np.dot(F , points1)))
        error.append(err)
    return err

#just to check using opencv function
def cv2F(points1 , points2):
    F , mask = cv2.findFundamentalMat(points1 , points2 ,cv2.RANSAC, 1, 0.90)
    u , s , vt = np.linalg.svd(F)
    s[2] = 0.0
    F = u.dot(np.diag(s).dot(vt))
    return F , mask

#same thing with error just nothing new


#trying to normalize points so that they have mean at origin
def normalize(points):
    mean = np.mean(points , axis = 0)
    xmean , ymean = mean[0] , mean[1]
    x , y = seperate(points)
    x , y = np.array(x) , np.array(y)
    xhat , yhat = x - xmean , y - ymean

    s = (2/np.mean(xhat**2 + yhat**2))**(0.5)
    T_s = np.diag([s,s,1])
    T_t = np.array([[1,0,-xmean] , [ 0,1,-ymean] , [0,0,1]])
    T = np.dot(T_s , T_t)
    x_ = np.column_stack((points , np.ones(len(points)))) 
    x_norm = (np.dot(T , x_.T)).T
    return x_norm , T

#get F from F_normalized
def getF(Fnorm , T2 , T1):
    Forg = T2.T @ Fnorm @ T1
    return Forg

# main ransac function which gives the fundamental matrix


