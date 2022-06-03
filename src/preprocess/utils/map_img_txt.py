import PIL.Image
import numpy as np
import pandas as pd

def map_img_txt(path_in = 'res/images_info_0.csv', path_out = 'res/images_info_1.csv'):
  
  ''' 
  
  This function maps the image paths with the features contained in .txt file
  
  Returns: df
  '''
  
  data = []
  x_coord = []
  y_coord = []
  size = []
  names = []
  radius = []
  image_path = []
  cat = []

  path_labels = path_in

  with open(path_labels) as file:
    next(file)
    for l in file:
      i = l.split(",")[1]
      images_path = 'res/all-mias/{:}.pgm'.format(i)
      img = PIL.Image.open(images_path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array(i))
      cat.append(np.array(l.split(",")[2]))
      x_coord.append(np.array(l.split(",")[3]))
      y_coord.append(np.array(l.split(",")[4]))
      radius.append(np.array((l.split(",")[5]).rstrip()))
      image_path.append(images_path)
  
  arr = np.array([names, data, size, cat, x_coord, y_coord, radius, image_path], dtype = object).T.tolist()
  df = pd.DataFrame(data = arr, columns = ['name', 'features', 'size', 'cat', 'x_coord', 'y_coord', 'radius', 'image path'])
  df['x_coord'].replace('','0',inplace=True)
  df['y_coord'].replace('','0',inplace=True)
  df['radius'].replace('','0',inplace=True)
  df['y_coord'] = int(1024) - df['y_coord'].astype(str).astype(float).astype(int)
  df['xmin'] = df['x_coord'].astype(str).astype(float).astype(int) - df['radius'].astype(str).astype(float).astype(int)
  df['xmax'] = df['x_coord'].astype(str).astype(float).astype(int) + df['radius'].astype(str).astype(float).astype(int)
  df['ymin'] = df['y_coord'].astype(str).astype(float).astype(int) - df['radius'].astype(str).astype(float).astype(int)
  df['ymax'] = df['y_coord'].astype(str).astype(float).astype(int) + df['radius'].astype(str).astype(float).astype(int)

  df.to_csv(path_out, sep = ",")

  return df