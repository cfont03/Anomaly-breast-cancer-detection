import cv2
import numpy as np
from pathlib import Path


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
  
    ### CHECKPOINT INPUT
    if (xcoord > size_x or ycoord > size_y or xmin > xmax or ymin > ymax):
        print("ERROR: coordinates are wrong!")
    else:
        pass
    
    vflip_ = cv2.imread(str(path))
    vflip = cv2.flip(vflip_, 0)
    new_path = Path("/content/archive/all-mias/{:}_vflip{:}".format(name, '.jpeg'))
    status = cv2.imwrite(str(new_path), vflip)

    ### CHECKPOINT OUTPUT
    print("Image written to file-system " , new_path,  " :", status)

    ### adapt bounding boxes points
    ycoord_ = int(size_y) - int(ycoord) # needs to be 1024 dynamic
    ymax_ = int(size_y) - int(ymin)
    ymin_ = int(size_y) - int(ymax)
    x_y.append(np.array([[xcoord, ycoord_]]))
    x_min_max.append(np.array([[xmin, xmax]]))
    y_min_max.append(np.array([[ymin_, ymax_]]))
  
  return x_y, x_min_max, y_min_max