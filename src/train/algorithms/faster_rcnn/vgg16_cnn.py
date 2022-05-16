import time
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras 
print("Keras version   : ", keras.__version__)
from tensorflow.keras import Model


def vgg16_cnn(df, h = 800, w = 800, c = 3):

  '''

  This function uses pre-trained vgg16 model to extract the feature map of the passed images. 

  Params: df with
  0: image path

  Returns: 
  - output model of vgg16
  - feature map
  - df with additional info:
  w_fm: width of the feature map
  h_fm: height of the feature map
  n_anchors: number of potential anchors per image

  '''

  # show execution time
  start_time = time.time()



  # --------------- vgg16 model -------------
   
  vgg16 = tf.keras.applications.VGG16(
    include_top=False, 
    weights='imagenet',
    input_shape = (w, h, c)
  )

  for layer in vgg16.layers:
    layer.trainable = True
  
  vgg16_model = Model(inputs= [vgg16.layers[0].input], outputs= [vgg16.layers[17].output])


  # train data
  train_images = []
  
  for i in df.iloc:
    img = cv2.imread(str(i[0]))
    train_images.append(img)

  train_images = np.array(train_images)
  train_images = train_images/255 # normalize images

  feature_map = vgg16_model.predict(train_images)


  # feature map
  anchors = []
  w_fms = []
  h_fms = []
  features = []

  for i in df.iloc:
    img = cv2.imread(str(i[0]))
    fm = vgg16_model.predict(np.expand_dims(img, 0))
    _, w_fm, h_fm, _ = fm.shape
    n_anchor = w_fm * h_fm  
    anchors.append(n_anchor)
    w_fms.append(w_fm)
    h_fms.append(h_fm)



  df['n_anchor'] = anchors
  df['w_fm'] = w_fms
  df['h_fm'] = h_fms 



  print(f"\n------- Execution time: {(time.time() - start_time)/60:.2f} minutes -------\n")
  print('Number of anchors needed: ', n_anchor)
  print('\n', vgg16_model.summary())
  
  
  return df, vgg16_model, feature_map