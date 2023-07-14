import numpy as np
import cv2
import argparse
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import csv

from Utils.read_data import *
from Utils.GetInliersRANSAC import *
from Utils.EstimateFundamentalMatrix import *
from Utils.EssentialMatrixFromFundamentalMatrix import *
from Utils.ExtractCameraPose import *
from Utils.DisambiguateCameraPose import *
from Utils.LinearTriangulation import *
from Utils.NonLinearTriangulation import *
from Utils.LinearPnP import *
from Utils.NonLinearPnP import *
from Utils.PnPRansac import *
from Utils.BuildVisibilityMatrix import *
from Utils.BundleAdjustment import *
from Utils.MiscUtils import *

Parser = argparse.ArgumentParser()
Parser.add_argument('--DataPath' , default ='./Data/' , help = 'floder where input data exists')
Parser.add_argument('--savepath' , default = './saved/' , help = 'folder to save all outputs')
Parser.add_argument('--load_data' , default= False, type = lambda x: bool(int(x)), help='load data from files or not')
Parser.add_argument('--BA' , default= True, type = bool, help='Do bundle adjustment or not')

Args = Parser.parse_args()
folder_name = Args.DataPath
savePath = Args.savepath
load_data = False
BA = bool(int(Args.BA))

bundle = True
saveimage = False
load = False
f = open('error_chart.csv', mode='w')
error_chart = csv.writer(f)
images = readImages(folder_name , 6)

K = np.array([[568.996140852, 0, 643.21055941],
              [0, 568.988362396, 477.982801038], 
              [0, 0, 1]])

# correspondences = get_correspondence()

# [ 12, 13, 14, 15, 16, 23, 24, 25, 26, 34, 35, 36, 45, 46, 56]

#extracting features from multiple images 
feature_x , feature_y , feature_flag , feature_descriptor = extractMatchingFeatures(folder_name , 6)
print(feature_flag.shape)
print(feature_descriptor.shape)


if load == True:
    print("Using saved f_matrix and feature flag")
    # filtered_feature_flag = np.load('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/filtered_feature_flag.npy' , allow_pickle = True)
    # f_matrix = np.load('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/f_matrix.npy',allow_pickle=True)
    filtered_feature_flag = np.load('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/filtered_feature_flag.npy' , allow_pickle = True)
    f_matrix = np.load('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/f_matrix.npy',allow_pickle=True)
    
else:
    filtered_feature_flag = np.zeros_like(feature_flag)
    f_matrix = np.empty(shape=(6 , 6) , dtype=object)
    for i in range(0,5):
        for j in range(i+1 , 6):
            idx = np.where(feature_flag[:,i] & feature_flag[:,j] )
            pts1 = np.hstack((feature_x[idx , i].reshape((-1,1)) , feature_y[idx , i].reshape((-1,1))))
            pts2 = np.hstack((feature_x[idx , j].reshape((-1,1)) , feature_y[idx , j].reshape((-1,1))))
            idx = np.array(idx).reshape(-1)
            # print(idx)
            if len(idx) > 8:
                F_best , chosen_idx = ransacInliers(pts1 , pts2 , idx)
                print('At image : ', i,j, '|| no of inliers: ', len(chosen_idx) , '/',len(idx))
                f_matrix[i,j] = F_best
                filtered_feature_flag[chosen_idx , j] = 1
                filtered_feature_flag[chosen_idx , i] = 1
    # np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/f_matrix' , f_matrix)
    # np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/filtered_feature_flag' , filtered_feature_flag)

print("doing images 1 and 2")
n , m = 0,1
F12 = f_matrix[n,m]
E12 = getEssentialMatrix(K , F12)
print("Estimating poses of camera 2")
R_set , C_set = decomposeE(E12)

# pts1_12 , pts2_12 = correspondence_updated[0][0] , correspondence_updated[0][1]
idx = np.where(filtered_feature_flag[:, n] & filtered_feature_flag[:,m])
pts1 =np.hstack((feature_x[idx , n].reshape((-1,1)), feature_y[idx , n].reshape((-1,1))))
pts2 =np.hstack((feature_x[idx , m].reshape((-1,1)), feature_y[idx , m].reshape((-1,1))))
# print('shape of pts1' ,pts1.shape)

R1_ = np.identity(3)
C1_ = np.zeros((3,1))
I = np.identity(3)

pts3D_4 = []
for i in range(len(R_set)):
    x1 = pts1
    x2 = pts2

    X = linearTriangulation(R1_ , C1_ , x1 , x2 , K , R_set[i] , C_set[i])
    X = X/X[:,3].reshape(-1,1)
    pts3D_4.append(X)
# print(np.array(pts3D_4).shape)
x,y,z = split(np.array(pts3D_4))

if saveimage == True:
    fig = plt.figure(figsize = (10,10))
    plt.xlim(-500 , 500)
    plt.ylim(-100 , 100)
    plt.scatter(x , z , marker='.' ,linewidths = 0.5 , color = 'blue')
    for i in range(4):
        R1 = getEuler(R_set[i])
        R1 = np.rad2deg(R1)
        plt.plot(C_set[i][0] , C_set[i][2] , marker = (3,0 ,int(R1[i])), markersize = 15 , linestyle = 'None')
    plt.savefig(savePath + 'lineartriangulation12.png')
    plt.show()


# print("shape of pts3D: " ,np.array(pts3D_4).shape)
# X = linearTriangulation(R_set , C_set , x1 , x2 , K , K)
# pts3D_4 = X

R_chosen , C_chosen , X = disambiguatePose(R_set , C_set , pts3D_4)
X = X/X[:,3].reshape(-1,1)
print("Done ###")
print("Performing nonlinear triangulation....")
X_refined = NonLinearTriangulation(K , pts1 , pts2 , X , R1_ , C1_ ,R_chosen , C_chosen )
X_refined = X_refined / X_refined[:,3].reshape(-1,1)

if saveimage == True:
# print(np.array(X_refined).shape)
    fig = plt.figure(figsize = (10,10))
    plt.xlim(-100 , 100)
    plt.ylim(-50 , 50)
    plt.scatter(x , z , marker='.' ,linewidths = 0.5 , color = 'blue')
    x , y , z = giveXYZ(np.array(X_refined))

    R1 = getEuler(R_chosen)
    R1 = np.rad2deg(R1)
    plt.plot(C_chosen[0] , C_chosen[2] , marker = (3,0 ,int(R1[1])), markersize = 15 , linestyle = 'None')
    plt.savefig(savePath + 'nonlineartriangulation12_1.png')
    plt.show()


mean_error1 = meanReprojectionError(X , pts1 , pts2 , R1_ , C1_ , R_chosen, C_chosen, K )
mean_error2 = meanReprojectionError(X_refined , pts1 , pts2 , R1_ , C1_ , R_chosen , C_chosen ,K)
print(n+1 , m+1 , 'Before optimization LT: ' , mean_error1 , 'After optimization: ',mean_error2)
print("Done ####")

error_row_chart = np.zeros((20))
error_row_chart[3] = mean_error1
error_row_chart[9] = mean_error2
error_chart.writerow(list(error_row_chart))

######### Register for Camera 1 and 2 #######
X_all = np.zeros((feature_x.shape[0] ,3))
camera_indices = np.zeros((feature_x.shape[0] , 1) , dtype = int)
X_found = np.zeros((feature_x.shape[0] , 1) , dtype = int)

X_all[idx] = X[:, :3] 
X_found[idx] = 1
camera_indices[idx] = 1

X_found[np.where(X_all[:,2] < 0 )] = 0

C_set_ = []
R_set_ = []

C0 = np.zeros(3)
R0 = np.identity(3)

C_set_.append(C0)
R_set_.append(R0)

C_set_.append(C_chosen)
R_set_.append(R_chosen)

print("Doing other images")


for i in range(2 , 6):
    error_row_chart = np.zeros((20))
    print("running for image: ", str(i+1) , '.....')

    feature_idx_i = np.where(np.logical_and( X_found[:, 0] , filtered_feature_flag[:,i] ))
    # print(feature_idx_i)
    if len(feature_idx_i[0]) < 8 :
        print("Found ",len(feature_idx_i), "common points between X and ",i," image")
        continue

    pts_i = np.hstack((feature_x[feature_idx_i , i].reshape(-1,1) , feature_y[feature_idx_i ,i].reshape(-1,1)))
    X = X_all[feature_idx_i , :].reshape(-1,3)

    # print("shape of X pnp",X.shape)
    # print("shape of x pnp" ,pts_i.shape)
    if X.shape[0] != pts_i.shape[0]:
        continue
    C_init ,R_init  = PnPRANSAC(X, pts_i  ,K)
    ('print done pnpransac')
    print("Rinit: ", R_init.shape)
    print("C_init: " , C_init.shape)
    errorLinearPnp = reprojErrorPnP(X , pts_i , K , R_init , C_init)

    Ri , Ci = NonLinearPnP(K ,pts_i , X , R_init , C_init)
    errorNonLinearPnP = reprojErrorPnP(X , pts_i , K , Ri , Ci)
    if saveimage == True:
        x , y , z = giveXYZ(X)
        fig = plt.figure(figsize = (10,10))
        plt.xlim(-100 , 100)
        plt.ylim(-100 , 100)
        plt.scatter(x , z , marker='.' ,linewidths = 0.5 , color = 'blue')
        R1 = getEuler(Ri)
        R1 = np.rad2deg(R1)
        plt.plot(Ci[0] , Ci[2] , marker = (3,0 ,int(R1[1])), markersize = 15 , linestyle = 'None')
        plt.savefig(savePath + 'PnP' + str(i+1) + '.png')
        plt.show()

    print("Error after linear Pnp: ", errorLinearPnp,"error after non linear Pn: ", errorNonLinearPnP)

    error_row_chart[0] = errorLinearPnp
    error_row_chart[1] = errorNonLinearPnP

    C_set_.append(Ci)
    R_set_.append(Ri)

    for j in range(0,i):
        idx_X_pts = np.where(filtered_feature_flag[: ,j] & filtered_feature_flag[:, i])
        if (len(idx_X_pts[0]) < 8 ):
            continue

        x1 = np.hstack((feature_x[idx_X_pts , j].reshape((-1,1)), feature_y[idx_X_pts , j].reshape((-1,1))))
        x2 = np.hstack((feature_x[idx_X_pts , i].reshape((-1,1)), feature_y[idx_X_pts , i].reshape((-1,1))))

        X = linearTriangulation(R_set_[j] , C_set_[j] , x1 , x2 , K , Ri , Ci)
        if saveimage == True:
            x11,y11,z11 = giveXYZ(X)
            # print(x11,z11)
            fig = plt.figure(figsize = (10,10))
            plt.xlim(-1 , 1)
            plt.ylim(0.5 , 1.5)
            plt.scatter(x11, z11 , marker='.' ,linewidths = 0.5 , color = 'green')
            for i in range(4):
                R1 = getEuler(R_set[i])
                R1 = np.rad2deg(R1)
                plt.plot(C_set[i][0] , C_set[i][2] , marker = (3,0 ,int(R1[i])), markersize = 15 , linestyle = 'None')
            plt.savefig(savePath + 'lineartriangulation'+ str(i) + str(j) + '.png')
            plt.show()
        

        LT_error = meanReprojectionError(X , x1 , x2 , R_set_[j] , C_set_[j], Ri  ,Ci  ,K )
        X = NonLinearTriangulation(K , x1 , x2 , X , R_set_[j] , C_set_[j] , Ri , Ci )
        X = X/X[:,3].reshape(-1,1)
        nLT_error = meanReprojectionError(X , x1 , x2 , R_set_[j] , C_set_[j] , Ri , Ci , K)

        error_row_chart[2 + j] = LT_error
        error_row_chart[8 + j] = nLT_error

        X_all[idx_X_pts] = X[:,:3]
        X_found[idx_X_pts] = 1
        print('appended' , len(idx_X_pts[0]), "points between ",j," and  " ,i)
    
    if bundle == True:
        print('Performing Bundle Adjustment  for image : ', i  )
        R_set_, C_set_, X_all = BundleAdjustment(X_all,X_found, feature_x, feature_y,
                                                filtered_feature_flag, R_set_, C_set_, K, nCam = i)
        print("Ba done")
           
        for k in range(0 , i+1):
            idx_X_pts = np.where(X_found[:,0] & filtered_feature_flag[:, k])
            x = np.hstack((feature_x[idx_X_pts, k].reshape((-1, 1)), feature_y[idx_X_pts, k].reshape((-1, 1))))
            X = X_all[idx_X_pts]
            BA_error = reprojErrorPnP(X, x, K, R_set_[k], C_set_[k])
            print("Error after BA :", BA_error)
            error_row_chart[14+k] = BA_error
 
    print( " done camera ", i +1)
    error_chart.writerow(list(error_row_chart))
    
X_found[X_all[:,2] < 0 ] = 0
# if bundle == True:
#     print("doing with bundlee adjustment")
#     np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/optimised_C_set',C_set_)
#     np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/optimised_R_set',R_set_)
#     np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/optimised_X_all',X_all)
#     np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/optimised_X_found',X_found)

# else:
#     print("printing without bundle")
#     np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/C_set_BBA',C_set_)
#     np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/R_set_BBA',R_set_)
#     np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/X_all_BBA',X_all)
#     np.save('/Users/sheriarty/Desktop/CMSC733/CMSC733Proj3/saved/X_found_BBA',X_found)


feature_idx = np.where(X_found[:, 0])
X = X_all[feature_idx]
x = X[:,0]
y = X[:,1]
z = X[:,2]

fig = plt.figure(figsize = (10,10))
plt.xlim(-10 , 25)
plt.ylim(-10 , 25)
plt.scatter(x , z , marker='.' ,linewidths = 0.5 , color = 'blue')
for i in range(0 , len(C_set_)):
    R1 = getEuler(R_set_[i])
    R1 = np.rad2deg(R1)
    plt.plot(C_set_[i][0] , C_set_[i][2] , marker = (3,0 ,int(R1[1])), markersize = 15 , linestyle = 'None')

plt.savefig(savePath + 'final2DBBA.png')
plt.show()

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection = "3d")

ax.scatter3D(x , y ,z , color = 'green')
plt.savefig(savePath+ 'final3DBBA.png')
plt.show()

f.close()

        
















