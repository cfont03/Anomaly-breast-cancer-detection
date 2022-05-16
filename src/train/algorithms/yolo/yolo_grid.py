import time
from tensorflow import keras 
print("Keras version   : ", keras.__version__)

import tensorflow as tf

from tensorflow.keras.layers import (Input,
     Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU)
from tensorflow.keras import Model
import tensorflow.keras.backend as K


def yolo_grid(c = 3, S = 7, C = 1, B = 1):

  '''

  This function implements yolo model architecture and outputs a feature vector of 7 x 7 x 30 per image passed, which corresponds to the S x S grid.


  '''

  # show execution time
  start_time = time.time()

  w = h = 448 # input size needs to be 448
  input_img = Input(shape = (w, h, c)) # define input image size
  true_box = Input(shape = (7,7,6))

  # ------ build neural network ------------

  # 24 convolutional layers

  # ---- feature extractor ----
  # block 1
  # layer1
  x = Conv2D(64, kernel_size = (7,7), strides = (2,2), padding = 'same', kernel_initializer = 'normal', name = 'cl1')(input_img)
  x = BatchNormalization(name='norm_1')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_1')(x)
  x = MaxPooling2D((2,2), strides = (2,2), name = 'mpl1')(x)

  # block 2
  # layer2
  x = Conv2D(192, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl2')(x)
  x = BatchNormalization(name='norm_2')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_2')(x)
  x = MaxPooling2D((2,2), strides = (2,2), name = 'mpl2')(x)

  # block 3
  # layer3
  x = Conv2D(128, kernel_size = (1,1), padding = 'same', kernel_initializer = 'normal', name = 'cl3')(x)
  x = BatchNormalization(name='norm_3')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_3')(x)
  # layer4
  x = Conv2D(256, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl4')(x)
  x = BatchNormalization(name='norm_4')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_4')(x)
  # layer5
  x = Conv2D(256, kernel_size = (1,1), padding = 'same', kernel_initializer = 'normal', name = 'cl5')(x)
  x = BatchNormalization(name='norm_5')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_5')(x)
  # layer6
  x = Conv2D(512, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl6')(x)
  x = BatchNormalization(name='norm_6')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_6')(x)
  x = MaxPooling2D((2,2), strides = (2,2), name = 'mpl3')(x)

  # block 4
  # layer7
  x = Conv2D(256, kernel_size = (1,1), padding = 'same', kernel_initializer = 'normal', name = 'cl7')(x)
  x = BatchNormalization(name='norm_7')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_7')(x)
  # layer8
  x = Conv2D(512, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl8')(x)
  x = BatchNormalization(name='norm_8')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_8')(x)
  # layer9
  x = Conv2D(256, kernel_size = (1,1), padding = 'same', kernel_initializer = 'normal', name = 'cl9')(x)
  x = BatchNormalization(name='norm_9')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_9')(x)
  # layer10
  x = Conv2D(512, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl10')(x)
  x = BatchNormalization(name='norm_10')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_10')(x)
  # layer11
  x = Conv2D(256, kernel_size = (1,1), padding = 'same', kernel_initializer = 'normal', name = 'cl11')(x)
  x = BatchNormalization(name='norm_11')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_11')(x)
  # layer12
  x = Conv2D(512, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl12')(x)
  x = BatchNormalization(name='norm_12')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_12')(x)
  # layer13
  x = Conv2D(256, kernel_size = (1,1), padding = 'same', kernel_initializer = 'normal', name = 'cl13')(x)
  x = BatchNormalization(name='norm_13')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_13')(x)
  # layer14
  x = Conv2D(512, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl14')(x)
  x = BatchNormalization(name='norm_14')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_14')(x)
  # layer15
  x = Conv2D(512, kernel_size = (1,1), padding = 'same', kernel_initializer = 'normal', name = 'cl15')(x)
  x = BatchNormalization(name='norm_15')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_15')(x)
  # layer16
  x = Conv2D(1024, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl16')(x)
  x = BatchNormalization(name='norm_16')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_16')(x)
  x = MaxPooling2D((2,2), strides = (2,2), name = 'mpl4')(x)

  # block 5
  # layer17
  x = Conv2D(512, kernel_size = (1,1), padding = 'same', kernel_initializer = 'normal', name = 'cl17')(x)
  x = BatchNormalization(name='norm_17')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_17')(x)
  # layer18
  x = Conv2D(1024, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl18')(x)
  x = BatchNormalization(name='norm_18')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_18')(x)
  # layer19
  x = Conv2D(512, kernel_size = (1,1), padding = 'same', kernel_initializer = 'normal', name = 'cl19')(x)
  x = BatchNormalization(name='norm_19')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_19')(x)
  # layer20
  x = Conv2D(1024, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl20')(x)
  x = BatchNormalization(name='norm_20')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_20')(x)
  
  # ---- object classifier -----
  # layer21
  x = Conv2D(1024, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl21')(x)
  x = BatchNormalization(name='norm_21')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_21')(x)
  # layer22
  x = Conv2D(1024, kernel_size = (3,3), strides = (2,2), padding = 'same', kernel_initializer = 'normal', name = 'cl22')(x)
  x = BatchNormalization(name='norm_22')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_22')(x)

  # block 6
  # layer23
  x = Conv2D(1024, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl23')(x)
  x = BatchNormalization(name='norm_23')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_23')(x)
  # layer24
  x = Conv2D(1024, kernel_size = (3,3), padding = 'same', kernel_initializer = 'normal', name = 'cl24')(x)
  x = BatchNormalization(name='norm_24')(x)
  x = LeakyReLU(alpha = 0.1, name = 'leaky_re_lu_24')(x)

  # FCN
  # layer1
  x = tf.keras.layers.Flatten()(x)
  x = Dense(4096)(x)
  x = Dropout(0.0)(x)
  x = LeakyReLU(alpha = 0.1)(x)
  
  # layer2
  x = Dense(S * S * (C + B * 5))(x)
  output = Lambda(lambda args: args[0])([x, true_box])

  model = Model(inputs = [input_img , true_box] , outputs = output)
  
  print('\n', model.summary())

  return model