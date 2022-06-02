import cv2
from pathlib import Path
import numpy as np
import os

def horizontal_flip (info_file, display = False):

  ''' 
  
  Function to apply a horizontal flip to the images.

  Args: df with Params:
    0: reference number
    1: features
    2: size
    3: class
    4: x_coordinate of the abnormality
    5: y_coordiante of the abnormality
    6: radius in pixels
    7: path of the image
    8: xmin coordinate bounding box
    9: ymin coordinate bounding box
    10: xmax coordinate bounding box
    11: ymax coordinate bounding box
  
  Outputs: saved imgs into system, list with (x,y) coordinates, list with (xmin, ymax) coordinates, list with (ymin, ymax) coordinates
  
  '''

  x_y = []
  x_min_max = []
  y_min_max = []

  for name, path, size_x, size_y, xcoord, ycoord, xmin, xmax, ymin, ymax in zip(info_file.iloc[:, 0], info_file.iloc[:, 7], 
                                                                                info_file.iloc[:, 2], info_file.iloc[:, 2],
                                                                                info_file.iloc[:, 4], info_file.iloc[:, 5],
                                                                                info_file.iloc[:, 8], info_file.iloc[:, 9],
                                                                                info_file.iloc[:, 10], info_file.iloc[:, 11]):
  

    hflip_ = cv2.imread(str(path))
    hflip = cv2.flip(hflip_, 1) 
    new_path = Path("res/all-mias/{:}_hflip{:}".format(name, '.jpeg')) 
    if os.path.exists(new_path) == True:
          print(new_path, ": File already exists")
    else:
          status = cv2.imwrite(str(new_path), hflip) 
          ### CHECKPOINT OUTPUT
          print("Image written to file-system " , new_path,  " :", status)

    ### adapt bounding boxes points
    xcoord_ = int(1024) - int(float(xcoord))
    xmax_ = int(1024) - int(xmin)
    xmin_ = int(1024) - int(xmax)
    x_y.append(np.array([[xcoord_, ycoord]]))
    x_min_max.append(np.array([[xmin_, xmax_]]))
    y_min_max.append(np.array([[ymin, ymax]]))
  
  return x_y, x_min_max, y_min_max