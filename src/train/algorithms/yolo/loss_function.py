import tensorflow.keras.backend as K


def yolo_sse (y_true, y_pred, S=7, B=1, C=1):

      ''' 

      This function implements the sum squared error loss function from the yolo model.  
      Assumption: only one box is predicted by image passed.

      Args: target boxes are in format [c, s, x, y, w, h], where c is the class type and s the confidence score.
            pred boxes are in format [s, x, y, w, h, c]

      Returns: loss value per image

      '''


      coord=5
      noobj=0.5

      true_boxes = K.reshape(y_true[...,2:6], (-1, S * S, B, 4))
      y_pred = K.reshape(y_pred, (-1, S, S, C + B * 5))
      pred_boxes = K.reshape(y_pred[...,1:5], (-1, S * S, B, 4))
      y_true_conf = K.reshape(y_true[...,1:2], (-1, S * S, B, 1))
      y_true_conf = y_true_conf[...,0]
      
      #----------- xy loss ----------
      # select xy coordinates
      y_pred_xy   = pred_boxes[...,0:2]
      y_true_xy   = true_boxes[...,0:2]
  
      # calculate xy_loss
      xy_loss = coord * K.sum(K.sum(K.square(y_true_xy - y_pred_xy),axis=-1)*y_true_conf, axis=-1)



      #---------- wh loss ------------
      # select wh values
      y_pred_wh   = pred_boxes[...,2:4]
      y_true_wh   = true_boxes[...,2:4]
      
      # calculate loss
      wh_loss = coord *  K.sum(K.sum(K.square(K.sqrt(y_true_wh) - K.sqrt(y_pred_wh)), axis=-1)*y_true_conf, axis=-1)
      

      #---------- class loss ----------
      # Since we are not interested in the class prediction, the class loss can be ignored
      #y_true_class = y_true[...,0:1]
      #y_true_class = y_true_class[...,0]
      #y_pred_class = y_pred[...,5:6]
      #y_pred_class = y_pred_class[0]
      #print(y_true_class.shape, y_pred_class.shape)
      #clss_loss  = K.sum(K.square(y_true_class - y_pred_class)*y_true_conf, axis=-1)

      #---------- confidence loss -----
      # calculate iou
      intersect_wh = K.maximum(K.zeros_like(y_pred_wh), (y_pred_wh + y_true_wh)/2 - K.abs(y_pred_xy - y_true_xy) )
      intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
      true_area = y_true_wh[...,0] * y_true_wh[...,1]
      pred_area = y_pred_wh[...,0] * y_pred_wh[...,1]
      union_area = pred_area + true_area - intersect_area
      iou = intersect_area / union_area
      
      y_pred_conf = K.reshape(y_pred[...,0:1], (-1, S * S, B, 1))
      y_pred_conf = y_pred_conf[...,0]
      conf_loss = K.sum(K.square(y_true_conf*iou - y_pred_conf), axis=-1) + K.sum(K.square(y_true_conf*iou - y_pred_conf), axis=-1) * noobj
      
      # ---------- total loss ----------
      loss = xy_loss + wh_loss + conf_loss
      loss = K.reshape(loss, (-1, 49, 1, 1))
      
      return loss