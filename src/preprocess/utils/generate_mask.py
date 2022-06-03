import cv2
import numpy as np
import os

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
   


    mask = np.zeros(shape=size, dtype=np.uint8)
    cv2.circle(mask, center = (int(float(x_coord)), int(float(y_coord))), radius = int(float(radius)), thickness = -1, color = (255, 255, 255))
    new_path = 'res/all-mias/{:}_mask.pgm'.format(name)
    if os.path.exists(new_path) == True:
          print(new_path, ": File already exists")
    else:
          status = cv2.imwrite(str(new_path), mask) 
          ### CHECKPOINT OUTPUT
          print("Image written to file-system " , new_path,  " :", status)
