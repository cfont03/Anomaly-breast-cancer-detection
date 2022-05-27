import numpy as np


def coord_labels (df):

  '''

  This function obtains an array with the coordinates of possible anchor boxes and its labels.

  Args: df containing:

  1: image path
  2: xmin coord
  3: ymin coord
  4: xmax coord
  5: ymax coord
  16: array with anchors in order x,y,w,h
  17: ground truths
  18: iou
  19: labels

  Returns: 
    array with shape (x, 22500, 4), where x corresponds to the number of instances (aka images) the df contains.

  '''

  coords = np.array(df.iloc[:,16])
  labels = np.array(df.iloc[:,19])
  offset_list = np.zeros((len(df.iloc[:,16]), len(df.iloc[:,16].values[0]), len(df.iloc[:,16].values[0][0])))                                                                     
  label_list = np.zeros((len(df.iloc[:,19]), len(df.iloc[:,19].values[0]), 1))
  
  # replace null values with anchor values
  layer1 = 0
  for i in coords:
    layer2 = 0
    for z in i:
      offset_list[layer1, layer2, :] = z
      
      layer2 += 1

    layer1 += 1

  
  # repeat process for labels
  layer1 = 0
  for i in labels:
    layer2 = 0
    for z in i:
      label_list[layer1, layer2, :] = z

      layer2 += 1
    
    layer1 += 1


  return offset_list, label_list