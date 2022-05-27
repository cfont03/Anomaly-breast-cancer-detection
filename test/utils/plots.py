import cv2
from pathlib import Path
import matplotlib.pylab as plt
from matplotlib import patches


def boundary_box (df):
  
  '''

  This function plots a given image with its boundary boxes.

  Args: df with Params:
  1: name image
  8: path image
  9: min coord x
  10: max coord x
  11: min coord y
  12: max coord y

  Outputs: plot given image with boundary box

  '''

  image = input("introduce image name: ") 
  path = '/content/archive/all-mias/{:}'.format(image)
  im = cv2.imread(path)
  
  # generate x, y, width and height
  xmin = df[df['image path'].apply(lambda x : x == str(Path(path)))].iloc[:,9]
  xmax = df[df['image path'].apply(lambda x : x == str(Path(path)))].iloc[:,10]
  ymin = df[df['image path'].apply(lambda x : x == str(Path(path)))].iloc[:,11]
  ymax = df[df['image path'].apply(lambda x : x == str(Path(path)))].iloc[:,12]
  
  xmin, xmax, ymin, ymax = (xmin.values[0], xmax.values[0], ymin.values[0], ymax.values[0])

  ### CHECKPOINT INPUT
  if (xmin > xmax or ymin > ymax):
      print("ERROR: coordinates are wrong!")
  else:
      pass

  width = xmax - xmin
  height = ymax - ymin

  # Create figure and axes
  fig, ax = plt.subplots()

  ax.imshow(im)

  rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='b', facecolor='none')

  ax.add_patch(rect)

  plt.show()




