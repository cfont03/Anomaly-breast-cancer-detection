import numpy as np


def non_max_suppression_fast(roi_sorted, score_sorted, pre_NMS_topN = 12000, threshold=0.7, n_train_post_nms = 2000):

  '''

  This function filters top N anchor boxes (ROI), eliminates highly overlapping ROIs and selects "n_train_post_nms" filtered proposals.

  Args: list of rois in 3D, scores in 2D.

  Returns: rois filtered with corresponding scores.

  '''
  
  roi_sorted_filtered, score_sorted_filtered = [], [] # generate empty lists to append results

  for img, score in zip(roi_sorted, score_sorted): # for each img
    # coordinates of the roi boxes
    x1 = img[:, 0]
    y1 = img[:, 1]
    x2 = img[:, 2]
    y2 = img[:, 3]
    
    # area of all roi
    # + 1 to prevent division by zero
    width_list = (x2 - x1) + 1 
    height_list = (y2 - y1) + 1
    area =  width_list * height_list

    # initialize the list of keep indexes	
    keep = []

    # generate np array from top selected rois
    roi_idx = np.array(range(pre_NMS_topN))
    # total number of ROIs to select after NMS
    n_train_post_nms = n_train_post_nms


    # keep looping while some indexes still remain in the indexes
    # list
    while roi_idx.size > 0:
        current_id = roi_idx[0]

        # add the current rio id to keep list
        keep.append(current_id)
        

        # find the intersection
        xx1 = np.maximum(x1[current_id], x1[roi_idx[1:]]) # maximum of x1 of current and all other x1 roi
        yy1 = np.maximum(y1[current_id], y1[roi_idx[1:]])
        xx2 = np.minimum(x2[current_id], x2[roi_idx[1:]])
        yy2 = np.minimum(y2[current_id], y2[roi_idx[1:]])

        ww = np.maximum(0., xx2 - xx1 +1)
        hh = np.maximum(0., yy2 - yy1 +1)

        inter = ww * hh

        # iou
        iou = inter/ (area[current_id] + area[roi_idx[1:]] - inter)

        # select boxes whose overlap is less than the threshold
        keep_idx = np.where(iou <= threshold)[0]
        
        # update the ROI index list(* note +1 to the indices list)
        roi_idx = roi_idx[keep_idx + 1]

    # select only "top n_train_post_nms" proposals (2000)
    keep = keep[: n_train_post_nms]

    img = img[keep]
    score = score[keep]
        
    roi_sorted_filtered.append(img)
    score_sorted_filtered.append(score)
    
  return roi_sorted_filtered, score_sorted_filtered