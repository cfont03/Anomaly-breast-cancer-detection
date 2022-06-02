import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path


def resize_input (df, alg='', h = 800, w = 800):
  
  '''

  Function returns resized images to specified size. Format: df
  Default is 800 x 800.

  Params:
  0: image path
  1: xmin
  2: xmax
  3: ymin
  4: ymax
  5: height original image
  6: width original image
  7: label
  
  Args: name for the algorithm
  
  Output: df with same structure as Args

  '''

  paths = []
  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  labels = []

  for i in df.iloc:

    img = cv2.imread(str(i[0]))
    img = cv2.resize(img, dsize = (h,w), interpolation = cv2.INTER_CUBIC)
    path, filename_ext = os.path.split(i[0])
    filename = os.path.basename(i[0]).split('.')[0]
    new_path = Path(str(path)+'/{:}_resize{:}{:}'.format(filename, alg, '.jpeg')) 
    if os.path.exists(new_path) == True:
          print(new_path, ": File already exists")
    else:
          status = cv2.imwrite(str(new_path), img) 
          ### CHECKPOINT OUTPUT
          print("Image written to file-system " , new_path,  " :", status)
  
    paths.append(new_path)

    x_factor = w / i[5]
    y_factor = h / i[6]

    
    xmin = i[1] * x_factor
    xmins.append(xmin)
    xmax = i[2] * x_factor
    xmaxs.append(xmax)
    ymin = i[3] * y_factor
    ymins.append(ymin)
    ymax = i[4] * y_factor
    ymaxs.append(ymax)

    labels.append(i[7])

  arr = np.array([paths, xmins, xmaxs, ymins, ymaxs, labels], dtype = object).T.tolist()
  df_ = pd.DataFrame(data = arr, columns = ['image path', 'xmin', 'xmax', 'ymin', 'ymax', 'labels'])
  
  return df_


