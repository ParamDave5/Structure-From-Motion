import numpy as np

#calcuate projection matrix
def projectionMatrix(K , R , C):
    I = np.identity(3)
    P = np.dot(K , np.dot(R , np.hstack((I , -C))))
    return P

#compute A matrix
def AMatrix(pt1,pt2,P1,P2):
    p1 , p2 , p3 = P1
    p11 , p12 , p13 = P2
    p1 , p2 , p3 = p1.reshape(1,-1) , p2.reshape(1,-1) ,p3.reshape(1,-1) 
    p11 , p12 , p13 = p11.reshape(1,-1) , p12.reshape(1,-1) ,p13.reshape(1,-1) 
    x,y = pt1
    x1 , y1 = pt2
    A = np.vstack((y*p3 - p2 , p1 - x*p3  , 
                    y1*p13 - p12 , p11 - x1*p13))
    return A

#triangulation 3d points
def triangulation(points1 , points2 , R1 , R2 , C1 , C2 , k1 , k2):
    P1 = projectionMatrix(k1 , R1 , C1)
    P2 = projectionMatrix(k2 , R2 , C2)
    point3d = []
    for p1,p2 in zip(points1 , points2):
        A = AMatrix(p1 , p2 , P1 , P2)
        u , s , vt = np.linalg.svd(A)
        v = vt.T
        x = v[:,-1]
        # x = x/x[-1]
        point3d.append(x)
    return np.array(point3d)

#linear triangulation
def LinearTriangulation(Rset , Cset , points1 , points2 , k1,k2):
    points3d = []
    for i in range(len(Rset)):
        R1 , R2 = np.identity(3) , Rset[i]
        C1 , C2 = np.zeros((3,1)) , Cset[i].reshape(3,1)
        point3d = triangulation(points1 , points2 , R1 , R2 , C1 , C2 , k1,k2)
        point3d = point3d/point3d[:,3].reshape(-1,1)
        points3d.append(point3d)
    return np.array(points3d)

def linearTriangulation(R1 , C1, points1,points2 , K , R2 , C2  ):
    I = np.identity(3)

    C1 = np.reshape(C1 ,(3,1))
    C2 = np.reshape(C2 ,(3,1))

    # print("dim of K" , K.shape)

    # print("dim of R1" , R1.shape)
    # print("dim of C1" , C1.shape)

    # print("dim of R2" , R2.shape)
    # print("dim of C2" , C2.shape)

    P1 = np.dot(K,np.dot(R1 , np.hstack((I,-C1))))
    P2 = np.dot(K,np.dot(R2 , np.hstack((I,-C2))))

    p1 = P1[0,:].reshape(1,4)
    p2 = P1[1,:].reshape(1,4)
    p3 = P1[2,:].reshape(1,4)

    p11 = P2[0,:].reshape(1,4)
    p12 = P2[1,:].reshape(1,4)
    p13 = P2[2,:].reshape(1,4)

    X = []

    for i in range(points1.shape[0]):
        x1 = points1[i,0]
        y1 = points1[i,1]

        x2 = points2[i,0]
        y2 = points2[i,1]

        A = []
        A.append((y1*p3) - p2)
        A.append((p1 - (x1*p3)))
        A.append((y2*p13) - p12)
        A.append((p11 - (x2*p13)))

        A = np.array(A).reshape(4,4)
        u , s , v = np.linalg.svd(A)
        v = v.T
        x = v[:,-1]
        X.append(x)
    return np.array(X)









