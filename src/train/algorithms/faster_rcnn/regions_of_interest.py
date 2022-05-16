import numpy as np


def rpn_to_roi (df, x_scores, dx, dy, dw, dh, threshold = 1):

  '''

  This function obtains the regional proposals (roi) based on the anchor boxes and the deltas obtained from the regional proposal network.

  Args: scores and deltas (dx, dw, dw, dh) and threshold, which will be used to eliminate anchor boxes exceeding it.

  df with:
  15: anchor boxes with xmin, xmax, ymin, ymax
  16: anchor boxes with x,y,w,h

  Returns:
  Roi in format x_min, y_min, x_max, y_max

  '''

  anchors_list = df.iloc[:,16]

  A = np.zeros((len(df), len(anchors_list[0]), 4)) # all images have the same anchors_list length (22500)
  
  layer1 = 0
  for anchors in anchors_list:
    layer2 = 0
    for x in anchors:
      anchor_centre_x = x[0]
      anchor_centre_y = x[1]
      anchor_width = x[2]
      anchor_height = x[3]

      # --------- adjust anchor boxes with predicetd offset-----------
      centre_x = dx[layer1][layer2] * anchor_width + anchor_centre_x
      centre_y = dy[layer1][layer2] *  anchor_height + anchor_centre_y
      width = np.exp(dw[layer1][layer2]) * anchor_width
      height = np.exp(dh[layer1][layer2]) * anchor_height

      # -------- conver them to min and max values -------------------
      x_min = centre_x - 0.5 * width
      y_min = centre_y - 0.5 * height
      x_max = centre_x + 0.5 * width
      y_max = centre_y + 0.5 * height

      # -------- add them to the empty array--------------------------
      A[layer1, layer2, :] = [x_min, y_min, x_max, y_max]

      layer2 +=1
    
    layer1 +=1


  A = A.T # transpose result

  
  roi  = np.clip(A, 0, 800) # clipping results to image size

  roi = roi.T # transpose result


  # ------- sort roi ascending order --------
  scores_sorted = np.zeros((498, 22500)) # 22500 scores per image
  rois_sorted = np.zeros((498, 22500, 4))
  sorted_idxs = np.zeros((498, 22500)) # 22500 indexes per image

  l = 0
  for img in (roi):
      img = np.int64(img)
      score = x_scores[l]
      sorted_idx = score.flatten().argsort()[::1] # generate ascending index
      score_sorted = score[sorted_idx] # sort the predicted scores
      roi_sorted = img[sorted_idx]

      rois_sorted[l, :] = roi_sorted
      scores_sorted[l, :] = score_sorted
      sorted_idxs[l, :] = sorted_idx.T

      l +=1


  return roi, scores_sorted, rois_sorted, sorted_idxs