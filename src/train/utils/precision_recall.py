import numpy as np


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