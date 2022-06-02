import numpy as np

from utils.intersection_over_union import IoU_ROI

def compute_precision_recall(df, thresholds):

    '''
    This function calculates the precision_recall curve

    '''

    precisions = []
    recalls = []
    # loop over each threshold
    for threshold in thresholds:
        
        _, idx_pos, idx_neg = IoU_ROI(df, pos_threshold = threshold, neg_threshold = 0.0001)
        
        tp = 0
        for i in idx_pos:
          if (len(i) > 0):
            tp += 1
          else:
            tp += 0

        fp = 0
        for i in idx_neg:
          if (len(i) > 0):
            fp += 1
          else:
            fp += 0

          
        fn = 498 - tp

        # compute precision and recall for each threshold
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
  
        # append precision and recall for each threshold to
        # precisions and recalls list
        precisions.append(np.round(precision, 3))
        recalls.append(np.round(recall, 3))

    return precisions, recalls


def compute_precision_recall_yolo(ious, thresholds):

    '''
    This function calculates the precision_recall curve

    '''

    precisions = []
    recalls = []

    for threshold in thresholds:
        
        scores = np.zeros((498, 7 * 7, 1))

        l1 = 0
        for image in ious:
          l2 = 0
          for grid in image:
            for iou in grid:
              if threshold <= iou <= 1:
                scores[l1, l2, :] = 2 # positive: FG
              elif 0 < iou < threshold:
                scores[l1, l2, :] = 1 # negative: BG
              else:
                pass
            l2 +=1
          l1 +=1

        
        
        tp = len(scores[scores == 2])
        fp = len(scores[scores == 1])
        fn = len(scores[scores == 0])

  
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
  
        precisions.append(np.round(precision, 3))
        recalls.append(np.round(recall, 3))

    return precisions, recalls