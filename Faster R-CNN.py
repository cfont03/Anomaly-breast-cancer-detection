# import of libraries
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from urllib import request
from tensorflow.keras import initializers
np.set_printoptions(suppress=True)
from pathlib import Path
from numpy.core.fromnumeric import shape


# resizing input images to fit into pretrained model, to extract features
def resize (df, h = 800, w = 800):
  
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
  6: weight original image
  7: label

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
    path, filename = os.path.split(i[0])
    new_path = Path(str(path)+'/{:}_resize{:}'.format(i[0][-10:-4], '.jpeg')) # generate new path
    status = cv2.imwrite(str(new_path), img)
    print("Image written to file-system " , new_path,  " :", status) # check if saved
    paths.append(new_path)

    x_factor = w / i[5]
    y_factor = h / i[6]

    # adapt bounding box accordingly
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
  df_ = pd.DataFrame(data = arr, columns = ['path', 'xmin', 'xmax', 'ymin', 'ymax', 'labels'])
  
  return df_
  
  
  
# show boundary box to resized image

def boundary_box_resize (df):
  
  '''

  This function plots an image with its boundary boxes

  Params:
  0: path image
  1: min coord x
  2: max coord x
  3: min coord y
  4: max coord y

  '''

  i = df.iloc[0]
  image = input("introduce image name: ")
  path, filename = os.path.split(i[0])
  im = cv2.imread(str(path) + '/{:}_resize.jpeg'.format(image))

  # generate x, y, width and height
  xmin, xmax, ymin, ymax = (i[1], i[2], i[3], i[4])
  width = xmax - xmin
  height = ymax - ymin

  # Create figure and axes
  fig, ax = plt.subplots()

  # Display the image
  ax.imshow(im)

  # Create a Rectangle patch
  rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')

  # Add the patch to the Axes
  ax.add_patch(rect)

  plt.show()
  
  
  
from numpy.core.fromnumeric import shape

# VGG Network
def vgg16_cnn(df, h = 800, w = 800):

  '''

  Returns a VGG16 model and a list of anchors needed per image passed

  Params: df with
  0: image path

  '''


  vgg = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet')

  input = vgg.input
  output = vgg.layers[-1].output

  vgg16_model = Model(inputs = input, outputs = output)
  
  
  # feature map
  anchors = []
  w_fms = []
  h_fms = []

  for i in df.iloc:
    img = cv2.imread(str(i[0]))
    shape_fm = vgg16_model.predict(np.expand_dims(img, 0))
    _, w_fm, h_fm, _ = shape_fm.shape
    n_anchor = w_fm * h_fm    # calculate number of anchors per image
    anchors.append(n_anchor)
    w_fms.append(w_fm)
    h_fms.append(h_fm)


  # add feature map in the df
  df['n_anchor'] = anchors
  df['w_fm'] = w_fms
  df['h_fm'] = h_fms


  return print('Number of anchors needed: ', n_anchor), vgg16_model.summary()
  


