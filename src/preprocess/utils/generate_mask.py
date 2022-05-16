import cv2
import numpy as np


def generate_mask (info_file, display = False):

  ''' 

  Function to generate masks for the images in the database 

  Args: df with Params:
    0: reference number
    1: features
    2: size
    3: class
    4: x_coordinate of the abnormality
    5: y_coordiante of the abnormality
    6: radius in pixels
    7: path of the image
  
  '''

  for name, size, x_coord, y_coord, radius in zip(info_file.iloc[:, 0], info_file.iloc[:, 2], 
                                                  info_file.iloc[:, 4], info_file.iloc[:, 5], info_file.iloc[:, 6]):
   
    ### CHECKPOIINT ARGS
    if (x_coord > size or y_coord > size):
      print("ERROR: coordinates are wrong!")
    else:
      pass

    mask = np.zeros(shape=size, dtype=np.uint8)
    cv2.circle(mask, center = (int(x_coord), int(y_coord)), radius = radius, thickness = -1, color = (255, 255, 255))
    new_path = '/content/archive/all-mias/{:}_mask.pgm'.format(name)
    status = cv2.imwrite(str(new_path), mask) 

    ### CHECKPOINT OUTPUT
    print("Image written to file-system " , new_path,  " :", status)