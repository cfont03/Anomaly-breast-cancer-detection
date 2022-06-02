import cv2
import numpy as np
from pathlib import Path
import os

def vertical_flip (info_file, display = False):

  ''' 
  
  Function to apply a vertical flip to the images

  Params:
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
  
    vflip_ = cv2.imread(str(path))
    vflip = cv2.flip(vflip_, 0)
    new_path = Path("res/all-mias/{:}_vflip{:}".format(name, '.jpeg'))
    if os.path.exists(new_path) == True:
          print(new_path, ": File already exists")
    else:
          status = cv2.imwrite(str(new_path), vflip)
          ### CHECKPOINT OUTPUT
          print("Image written to file-system " , new_path,  " :", status)

    ### adapt bounding boxes points
    ycoord_ = int(1024) - int(float(ycoord))
    ymax_ = int(1024) - int(ymin)
    ymin_ = int(1024) - int(ymax)
    x_y.append(np.array([[xcoord, ycoord_]]))
    x_min_max.append(np.array([[xmin, xmax]]))
    y_min_max.append(np.array([[ymin_, ymax_]]))
  
  return x_y, x_min_max, y_min_max