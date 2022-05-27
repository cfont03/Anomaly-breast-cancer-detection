import numpy as np


def coord_cells (df):

  '''

  This function obtains an array with the coordinates of the boxes relative to the image and relative to the cell

  Args: df containing:

  0: image path
  1: true values relative to the image
  2: true values relative to the cell

  Returns: 
    two arrays

  '''

  img_values = np.array(df.iloc[:,1])
  cell_values = np.array(df.iloc[:,2])

  img_list = np.zeros((len(df.iloc[:,1]), 1, len(df.iloc[:,1].values[0])))
  cell_list = np.zeros((len(df.iloc[:,2]), len(df.iloc[:,2].values[0]), len(df.iloc[:,2].values[0][0]), len(df.iloc[:,2].values[0][0][0]))) # generate empty np                                                                                                  
  
  # replace null values with values
  layer1 = 0
  for i in img_values:
    img_list[layer1, :] = i
    
    layer1 += 1

  
  # repeat process for labels
  layer1 = 0
  for i in cell_values:
    layer2 = 0
    for z in i:
      cell_list[layer1, layer2, :] = z

      layer2 += 1
    
    layer1 += 1


  return img_list, cell_list