import time
from tensorflow.keras import Model
from tensorflow import keras
import tensorflow.keras.backend as K
print("Keras version   : ", keras.__version__)
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D)
from tensorflow.keras.optimizers import Adam
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


def rpn (base_layer, w = 50, h = 50, c = 512):

  '''

  This function calculates the scores of each anchor box, as well as the deltas 

  '''

  # show execution time
  start_time = time.time()


  input_img = Input(shape = (w, h, c))
  
  #vgg16 = base_layer(input_img, training = True)
  
  x = Conv2D(512, kernel_size = (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(input_img) # conv layer with filter 3x3

  x_deltas = Conv2D(9 * 4, kernel_size = (1, 1), activation='linear',  
                                        kernel_initializer='zero', name='rpn_out_regress')(x) # conv layer for regression

  x_scores = Conv2D(9 * 1, kernel_size = (1, 1), activation='sigmoid', ## 9 * 1, FG (and BG)
                   kernel_initializer='uniform', name='rpn_out_score')(x) # conv layer for classification (whether the box is an object (FG) or background (BG))
  

  
  model = Model(inputs = [input_img] , outputs = [x_deltas, x_scores])
  
  model.summary()

  
  def smooth_l1_loss(y_true, y_pred):
    
    '''
    
    Calculates Smooth L1 loss
    
    '''

    
    x = K.abs(y_true - y_pred)

    # Find indices of values less than 1
    mask = K.cast(K.less(x, 1.0), "float32")
    # Loss calculation for smooth l1
    loss = (mask * (0.5 * x ** 2)) + (1 - mask) * (x - 0.5)
    return loss


  def custom_l1_loss(y_true, y_pred):
    
    '''

    Regress anchor offsets (deltas): only consider FG boxes
    
    '''

    offset_list= y_true[:,:,:-1]
    label_list = y_true[:,:,-1]
    
    # reshape output by the model
    y_pred = tf.reshape(y_pred, shape= (-1, 22500, 4))

    positive_idxs = tf.where(K.equal(label_list, 1)) # select only foreground boxes

    # Select positive predicted bbox shifts
    bbox = tf.gather_nd(y_pred, positive_idxs)
    
    target_bbox = tf.gather_nd(offset_list, positive_idxs)
    loss = smooth_l1_loss(target_bbox, bbox)

    return K.mean(loss)


  def custom_binary_loss(y_true, y_pred_objectiveness):
    
    '''
    
    Select both FG and BG class and compute cross entropy
    
    '''
    
    y_pred = tf.reshape(y_pred_objectiveness, shape= (-1, 22500))
    y_true = tf.squeeze(y_true, -1)
    
    # Find indices of positive and negative anchors, not neutral
    indices = tf.where(K.not_equal(y_true, -1)) # ignore -1 labels

    # Pick rows that contribute to the loss and filter out the rest.
    rpn_match_logits = tf.gather_nd(y_pred, indices)
    anchor_class = tf.gather_nd(y_true, indices)
    
    
    # Cross entropy loss
    loss = K.binary_crossentropy(target=anchor_class,
                                output=rpn_match_logits
                                )
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    
    return loss



  optimizer = Adam(learning_rate=1e-5)
  
  model.compile(optimizer=optimizer, loss=[custom_l1_loss, custom_binary_loss], run_eagerly=True) # use computed losses
  

  print(f"\n------- Execution time: {(time.time() - start_time)/60:.2f} minutes -------\n")

  return model, x_deltas, x_scores