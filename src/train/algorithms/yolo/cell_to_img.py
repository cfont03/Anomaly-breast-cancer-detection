import numpy as np
import keras.backend as K


def cell_to_img(y_pred, S = 7, B = 1, C = 1):

  '''

  This function takes two boundary boxes relative to cell grid and converts them relative to the image

  Args: array of size (x, S, S, B * 5 + C) where x stands for the length of the array

  Returns: np array of size (x, S * S, 4) where x stands for the length of the array
  
  '''

  w_img = h_img = 448
  batch_size = y_pred.shape[0]
  y_pred = K.reshape(y_pred, (batch_size, S, S, C + B * 5))

  bboxes = y_pred[...,2:6] 
  x, y, w, h = bboxes[...,0:1], bboxes[...,1:2], bboxes[...,2:3], bboxes[...,3:4] 
  s = y_pred[...,1:2]

  #l = np.array(( x - w/2) * w_img, dtype = int)
  #r = np.array(( x + w/2) * w_img, dtype = int)
  #t = np.array(( y - h/2) * h_img, dtype = int)
  #b = np.array(( y + h/2) * h_img, dtype = int)

  l = np.array(x * w_img + 448/2, dtype = int)
  r = np.array(y * h_img + 448/2, dtype = int)
  t = np.array(w * w_img, dtype = int)
  b = np.array(h * h_img, dtype = int)
  

  #l[l < 0] = l[l < 0] + 448/2
  #r[r > w_img] = w_img - 1
  #t[t < 0] = t[t < 0] * -1
  #b[b < 0] = b[b < 0] * -1

  y_pred_img = np.concatenate((s, l, r, t, b), axis = -1)
  
  return y_pred_img