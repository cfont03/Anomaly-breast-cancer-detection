import cv2
import numpy as np
import os

def rotate_images (info_file, display = False):

  ''' 

  Function to rotate images for the files in the database 

  Params:
    0: reference number
    1: features
    2: size
    3: class
    4: x_coordinate of the abnormality
    5: y_coordiante of the abnormality
    6: radius in pixels
    7: path of the image
    8: xmin
    9: ymin
    10: xmax
    11: ymax
  
  Outputs: saved imgs into system, list with (x,y) coordinates, list with (xmin, ymax) coordinates, list with (ymin, ymax) coordinates

  '''

  x_y = []
  x_min_max = []
  y_min_max = []

  for name, path, xcoord, ycoord, xmin, ymin, xmax, ymax in zip(info_file.iloc[:, 0], info_file.iloc[:, 7], 
                                              info_file.iloc[:, 4], info_file.iloc[:, 5],
                                              info_file.iloc[:, 8], info_file.iloc[:, 9],
                                              info_file.iloc[:, 10], info_file.iloc[:, 11]):
    
    
    img = cv2.imread(path) 
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    rot = cv2.getRotationMatrix2D((cX, cY), 45, 1)
    rotated = cv2.warpAffine(img, rot, (w, h))
    
    # rotate points accordingly
    xy = np.array([[int(float(xcoord)), int(float(ycoord))]])
    ones = np.ones(shape=(len(xy), 1))
    xy_ones = np.hstack([xy, ones])
    xy_points = rot.dot(xy_ones.T).T
    x_y.append(xy_points)

    # xmin_xmax
    x = np.array([[int(xmin), int(xmax)]])
    ones = np.ones(shape=(len(x), 1))
    x_ones = np.hstack([x, ones])
    x_points = rot.dot(x_ones.T).T
    x_min_max.append(x_points)

    # ymin_ymax
    y = np.array([[int(ymin), int(ymax)]])
    ones = np.ones(shape=(len(y), 1))
    y_ones = np.hstack([y, ones])
    y_points = rot.dot(y_ones.T).T
    y_min_max.append(y_points)


    new_path = 'res/all-mias/{:}_rotate.jpeg'.format(name)
    if os.path.exists(new_path) == True:
          print(new_path, ": File already exists")
    else:
          status = cv2.imwrite(str(new_path), rotated)
          ### CHECKPOINT OUTPUT
          print("Image written to file-system " , new_path,  " :", status)


  return x_y, x_min_max, y_min_max