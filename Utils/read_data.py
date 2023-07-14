import numpy as np
import cv2
import glob

# filename = 'Data/matching1.txt'
def find_matches(filename, image_no):
    correspondence_list = []
    correspondence_list1 = []
    with open(filename) as f:
        line_no = 0
        for line in f:
            line_no = line_no + 1
            if line_no == 1:
                continue
            l = line.split()
            for i in range(int(l[0]) - 1):
                if int(l[6+ 3*i]) == image_no:

                    correspondence_list1.append([float(l[4]) , float(l[5])])
                    correspondence_list.append([float(l[6+3*i + 1]), float(l[6+3*i + 2])])
    return [np.array(correspondence_list1),np.array(correspondence_list)]

def get_correspondence():
    I12= find_matches('Data/matching1.txt', 2)
    I13= find_matches('Data/matching1.txt', 3)
    I14= find_matches('Data/matching1.txt', 4)
    I15= find_matches('Data/matching1.txt', 5)
    I16= find_matches('Data/matching1.txt', 6)
    I23= find_matches('Data/matching2.txt', 3)
    I24= find_matches('Data/matching2.txt', 4)
    I25= find_matches('Data/matching2.txt', 5)
    I26= find_matches('Data/matching2.txt', 6)
    I34= find_matches('Data/matching3.txt', 4)
    I35= find_matches('Data/matching3.txt', 5)
    I36= find_matches('Data/matching3.txt', 6)
    I45= find_matches('Data/matching4.txt', 5)
    I46= find_matches('Data/matching4.txt', 6)
    I56= find_matches('Data/matching5.txt', 6)

    return [I12, I13, I14, I15, I16, I23, I24, I25, I26, I34, I35, I36, I45, I46, I56]

def readImages(folder , images):
    image = []
    for file in glob.glob(folder + "*.jpg"):
        file = cv2.imread(file)
        image.append(file)
    return image


def extractMatchingFeatures(folder , total_images):
    print('folder ', folder)
    feature_x = []
    feature_y = []
    feature_flag = []
    feature_descriptor = []

    for i in range(1 , total_images):
        file_name = folder + "matching" + str(i) + ".txt"
        file = open(file_name , 'r')
        for j , line in enumerate(file):
            if j == 0 :
                line_element = line.split(':')
                nFeatures = int(line_element[1])
            else:
                x_row = np.zeros((1,total_images))
                y_row = np.zeros((1,total_images))
                flag_row = np.zeros((1 , total_images) , dtype = int)

                line_element = line.split()
                features = [float(x) for x in line_element]
                features = np.array(features)

                n_matches = features[0]
                r = features[1]
                g = features[2]
                b = features[3]

                feature_descriptor.append([r,g,b])
                src_x = features[4]
                src_y = features[5]

                x_row[0 , i-1] = src_x
                y_row[0 , i-1] = src_y
                flag_row[0, i-1] = 1

                m = 1
                while n_matches > 1:
                    image_id = int(features[5+m])
                    image_id_x = features[6+m]
                    image_id_y = features[7+m]
                    m = m+3
                    n_matches = n_matches - 1
                    
                    x_row[0 , image_id - 1] = image_id_x
                    y_row[0 , image_id - 1] = image_id_y
                    flag_row[0 , image_id - 1] = 1
 
                feature_x.append(x_row)
                feature_y.append(y_row)
                feature_flag.append(flag_row)
    feature_x = np.array(feature_x).reshape(-1 , total_images)
    feature_y = np.array(feature_y).reshape(-1 , total_images)
    feature_flag =  np.array(feature_flag).reshape(-1 , total_images)
    feature_descriptor =  np.array(feature_descriptor).reshape(-1,3)
    return feature_x , feature_y , feature_flag , feature_descriptor
    




