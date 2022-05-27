import tensorflow as tf
import numpy as np


def IoU_faster_rcnn (df, threshold = 0.5):
    
    '''

    Calculate Intersection of Union (IoU) between anchor box and ground truth. 

    Params: df containing info on images
    1: image path
    2: x min coord
    3: x max coord
    4: y min coord
    5: y max coord
    6: label
    7: number of anchors
    8: width feature map
    9: height feature map
    10: centre x
    11: centre y
    12: centre list
    13: number of possible anchors
    14: number of objects
    15: anchor: predicted anchors in arrays. Sorted by xmax, ymax, xmin, ymin
    16: anchor: predicted anchors in arrays. Sorted by x, y, w, h
    17: boundary box: ground truth in array. Sorted by xmax, ymax, xmin, ymin

    Returns: An additional column in passed df with a list of arrays with the obtained IoUs: one IoU value per anchor box. 
    
    '''
    

    
    # ------calculate coordinate of overlapping region------

    truths = df.iloc[:,17]
    anchors = df.iloc[:,15]

    ious, labels = [], []

    for i in range(len(df)):
      
      ious_, labels_ = [], []
      
      for anchor in anchors[0]:

        x1 = max(truths[i][0], anchor[0])
        y1 = max(truths[i][1], anchor[1])
        x2 = min(truths[i][2], anchor[2])
        y2 = min(truths[i][3], anchor[3])
        
          
        # ------area of overlapping region------
        if (x2 < x1 and y2 < y1):
          area_overlap =  0
        else:
          width_overlap = (x2 - x1)
          height_overlap = (y2 - y1)
          area_overlap = width_overlap * height_overlap

          
        # ------computing union------
        width_truth = (truths[i][2] - truths[i][0])
        height_truth = (truths[i][3] - truths[i][1])

        width_anchor = (anchor[2] - anchor[0])
        height_anchor = (anchor[3] - anchor[1])
            
        
        area_truth = width_truth * height_truth
        area_anchor = width_anchor * height_anchor
            
        
        area_union_overlap = area_truth + area_anchor
        area_union = area_union_overlap - area_overlap
            
        iou = area_overlap/ area_union
        if iou < 0:
          ious_.append(0)
        else:
          ious_.append(iou)
        

        # ---------- add labels wrt iou value ---------

        if iou >= threshold:
          labels_.append(1)
        else:
          labels_.append(0)
        


      labels.append(labels_)
      ious.append(ious_)

    
    df['iou'] = ious
    df['labels_anchors'] = labels



    return df



def IoU_ROI (df, pos_threshold = 0.5, neg_threshold = 0.1):
    
    '''

    Calculate Intersection of Union (IoU) between ROI and ground truth. 

    Params: df containing info on images
    1: image path
    2: x min coord
    3: x max coord
    4: y min coord
    5: y max coord
    6: label
    7: number of anchors
    8: width feature map
    9: height feature map
    10: centre x
    11: centre y
    12: centre list
    13: number of possible anchors
    14: number of objects
    15: anchor: predicted anchors in arrays. Sorted by xmax, ymax, xmin, ymin
    16: anchor centre: predicted anchors in arrays. Sorted by x, y, w, h
    17: bbox: ground truth in array. Sorted by xmax, ymax, xmin, ymin
    18: iou between predicted anchors and ground truth
    19: labels based on iou in column 18
    20: roi filtered by top N values
    21: score per roi filtered by top N values

    max_iou: boundary of IOU: 50% by default
    min_iou: min. boundary for IOU: 10% by default

    Returns: An additional column in passed df with a list of arrays with the obtained IoUs. An extra column with the avg IoU per image
    and it also prints the avg IoU of the model.
    
    '''
    

    
    # ------calculate coordinate of overlapping region------

    truths = df.iloc[:,17]
    rois = df.iloc[:,20]

    ious, idx_pos, idx_neg = [], [], []

    for i in range(len(df)):
      
      ious_, idx_pos_, idx_neg_ = [], [], []
      idx = 0
      
      for roi in rois[i]:

        x1 = max(truths[i][0], roi[0])
        y1 = max(truths[i][1], roi[1])
        x2 = min(truths[i][2], roi[2])
        y2 = min(truths[i][3], roi[3])
        
          
        # ------area of overlapping region------
        if (x2 < x1 and y2 < y1): 
          area_overlap =  0
        else:
          width_overlap = (x2 - x1)
          height_overlap = (y2 - y1)
          area_overlap = width_overlap * height_overlap

          
        # ------computing union------
        width_truth = (truths[i][2] - truths[i][0])
        height_truth = (truths[i][3] - truths[i][1])

        width_roi = (roi[2] - roi[0])
        height_roi = (roi[3] - roi[1])
            
        area_truth = width_truth * height_truth
        area_roi = width_roi * height_roi

        area_union_overlap = area_truth + area_roi
        area_union = area_union_overlap - area_overlap
        iou = area_overlap/ area_union

        if iou >= pos_threshold:
          idx_pos_.append(idx)
        elif (iou < pos_threshold and iou >= neg_threshold):
          #ious_.append(iou)
          idx_neg_.append(idx)
        else:
          pass
        idx += 1

      idx_pos.append(idx_pos_)
      idx_neg.append(idx_neg_)

    df['iou_idx_pos'] = idx_pos
    df['iou_idx_neg'] = idx_neg


    return df, idx_pos, idx_neg




def IoU_yolo(box1, box2):

  ''' 
    
  This function calculates the Intersection Over Union between two passed boxes.

  Args:
  box1: first box to compare with [c,x,y,w,h] c = 0:1, x = 1:2, y = 2:3, w = 3:4, h = 4:5 y_true
  box2: second box to compare with [s, x,y,w,h] s = 0:1, x = 1:2, y = 2:3, w = 3:4, h = 4:5 y_pred
  
  Returns a value between 0 and 1

  '''

  # ------calculate coordinate of overlapping region------
  
  # calculate xmin, xmax, ymin and ymax
  xmin_b1 = box1[..., 1:2] - (box1[..., 3:4])/2
  ymin_b1 = box1[..., 2:3] - (box1[..., 4:5])/2
  xmax_b1 = box1[..., 1:2] + (box1[..., 3:4])/2
  ymax_b1 = box1[..., 2:3] + (box1[..., 4:5])/2

  xmin_b2 = tf.cast(box2[..., 1:2], tf.float64) - (box2[..., 3:4])/2
  ymin_b2 = tf.cast(box2[..., 2:3], tf.float64) - (box2[..., 4:5])/2
  xmax_b2 = tf.cast(box2[..., 1:2], tf.float64) + (box2[..., 3:4])/2
  ymax_b2 = tf.cast(box2[..., 2:3], tf.float64) + (box2[..., 4:5])/2
  
  
  # take max of x1 and y1 out of both boxes
  x1 = np.maximum(xmin_b1, xmin_b2)
  y1 = np.maximum(ymin_b1, ymin_b2)
  # take min of x2 and y2 out of both boxes
  x2 = np.maximum(xmax_b1, xmax_b2)
  y2 = np.maximum(ymax_b1, ymax_b2)

  
  # ------area of overlapping region------
  if (x1[0] < x2[0] and y1[0] < y2[0]):
    area_overlap = 0
  else:
    width_overlap = x2 - x1
    height_overlap = y2 - y1
    area_overlap = width_overlap * height_overlap
  
  

  # ------computing union------
  # sum of area of both the boxes - area_overlap
            
  # height and width of both boxes
  width_b1 = box1[...,3:4]
  height_b1 = box1[...,4:5]
  width_b2 = box2[...,3:4]
  height_b2 = box2[...,4:5]

  area_b1 = tf.cast((width_b1 * height_b1), tf.float64)
  area_b2 = tf.cast((width_b2 * height_b2), tf.float64)
  
  area_union_overlap = area_b1 + area_b2

  area_union = area_union_overlap - area_overlap
            
  iou = area_overlap/(area_union + 1e-6)

  return iou