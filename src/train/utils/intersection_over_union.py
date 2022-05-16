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


