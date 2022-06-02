import cv2
import numpy as np


def boundary_box_yolo(df, S = 7, B = 2, C = 20):
    
  ''' 
  This function adds two boundary boxes relative to image and to cell grid, as well as label matrix.

  Args: df with following params:
  0: image path
  1: x min of boundary box
  2: x max of boundary box
  3: y min of boundary box
  4: y max of boundary box

  Returns: df with added columns in [c, x, y, w, h] format

  '''

  # ---- boundary box relative to image -----
  bbox_img, bbox_cells = [], []
  
  for z in df.iloc: # for each image
    img = cv2.imread(str(z[0])) # read image
    size_w, size_h, _ = img.shape
    x = (z[2] + z[1])/2
    y = (z[4] + z[3])/2
    w = z[2] - z[1]
    h = z[4] - z[3]
    

    bbox_img.append([1, x, y, w, h]) # we only have 1 class to classify
    
    x_ = float(x / size_h)
    y_ = float(y / size_w)

    
  # ---- add bbox relative to cell grid --------
  for box in bbox_img:
    label_matrix = np.zeros((S, S, C + B * 5))
    c, x, y, w, h = box[0], box[1], box[2], box[3], box[4]

    i, j = int(S * y_), int(S * x_) # i, j represent the cell row and column positions the label belongs to
    x_cell, y_cell = x / size_w, y / size_h
    w_cell, h_cell = w / size_w, h / size_h
  
    if label_matrix[i, j, C] == 0: # if no value has been found for i,j ==> restricted to one object per cell
      label_matrix[i, j, C] = 1
      bbox_cell = [x_cell, y_cell, w_cell, h_cell] 
      label_matrix[i, j, C+1:C+5] = bbox_cell
      label_matrix[i, j, c] = 1
    
    
    bbox_cells.append(label_matrix)

  df['bbox_cxywh_img'] = bbox_img
  df['bbox_cywh_cell'] = bbox_cells

  df = df.iloc[:,[0,6,7]]

  return df