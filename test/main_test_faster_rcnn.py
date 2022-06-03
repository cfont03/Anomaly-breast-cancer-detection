import pandas as pd
import numpy as np
import keras


if __name__ == "__main__":
    from utils.resize_inputs import resize_input
    from utils.anchor_boxes import anchor_boxes
    from utils.anchor_points import anchor_points
    from utils.sample_anchors_pre import sample_anchors_pre
    from utils.precision_recall import compute_precision_recall
    from utils.non_max_supression import non_max_suppression_fast
    from utils.plots import boundary_box, plot_map
    from utils.intersection_over_union import IoU_faster_rcnn, IoU_ROI
    from utils.coordinate_labels import coord_labels
    from utils.vgg16_cnn import vgg16_cnn
    from utils.regions_of_interest import rpn_to_roi
else:
    pass

test = pd.read_csv('outputs/images_preprocess_test.csv', sep = ";")

# plot selected image
boundary_box(test)

test_ = test.iloc[:, [8, 9, 10, 11, 12]]
test_['height'] = 1024
test_['weight'] = 1024
test_['cat'] = test['cat']


### Adapt to fit VGG16
# resize to 800 x 800
test_re = resize_input(test_, w = 800, h = 800)

# calculate feature map
test_re, vgg16_model_test, feature_map_test = vgg16_cnn(test_re)


### CHECKPOINT
  ## Save files
test_re.to_csv("outputs/test_re.csv", sep = ";")
vgg16_model_test.save('outputs/vgg16_model_test.h5')
np.save('outputs/feature_map_test', feature_map_test)
  ## Upload files
vgg16_model_test = keras.models.load_model('outputs/vgg16_model_test.h5', compile = False)
test_re = pd.read_csv('outputs/test_re.csv', sep = ";")
feature_map_test = np.load('outputs/feature_map_test.npy')


### Anchor points and boxes
anchor_points(test_re)
anchor_boxes(test_re)
IoU_faster_rcnn(test_re)
sample_anchors_pre(test_re)
offset_list, label_list = coord_labels(test_re)

offset_list_label_list = np.zeros((204, 22500, 5))

layer1 = 0
for i in offset_list:
  for z in label_list:
    offset_list_label_list[layer1, :] = np.column_stack((i, z))[np.newaxis,:]
  
  layer1 +=1

### Regional Proposal Network
rpn_model_train = keras.models.load_model('outputs/rpn_model_vgg16.h5', compile = False)

### Predictions
pred_x_deltas, pred_x_scores = rpn_model_train.predict(feature_map_test)

n_anchors = 22500
pred_x_deltas = pred_x_deltas.reshape(-1, n_anchors, 4) 
pred_x_scores = pred_x_scores.reshape(-1, n_anchors)

dx = pred_x_deltas[:, :, 0]
dy = pred_x_deltas[:, :, 1]
dw = pred_x_deltas[:, :, 2]
dh = pred_x_deltas[:, :, 3]

## CHECKPOINT
np.save('outputs/pred_x_scores_test', pred_x_scores)
np.save('outputs/pred_x_deltas_test', pred_x_deltas)

### Regions of Interest
roi, scores_sorted, rois_sorted, sorted_idxs = rpn_to_roi(test_re, pred_x_scores, dx, dy, dw, dh)
roi_sorted_filtered, score_sorted_filtered = non_max_suppression_fast(rois_sorted, scores_sorted)
roi_sorted_filter = np.array(roi_sorted_filtered)
score_sorted_filter = np.array(score_sorted_filtered)

test_re['roi_filter'] = roi_sorted_filter
test_re['score_filter'] = score_sorted_filter

test_re, idx_pos, idx_neg = IoU_ROI(test_re)


### Model evaluation
rois_pos = []
for i, z in zip(test_re['roi_filter'], test_re['iou_idx_pos']):
  roi_pos = i[z]
  rois_pos.append(roi_pos)

test_re['rois_pos'] = rois_pos

# count positive and negative ious
pos_ious = 0
for i in test_re.iloc[:,-1]:
  if(len(i) > 0):
    pos_ious += 1
  else:
    pos_ious += 0

neg_ious = 0
for i in test_re.iloc[:,-2]:
  if(len(i) > 0):
    neg_ious += 1
  else:
    neg_ious += 0

all_ious = 0
for i in test_re.iloc[:,20]:
  all_ious += 1

print('TP: Positive predicted values (foregrounds): ', pos_ious, '/', f'{pos_ious / all_ious: .2%}')
print('FP: Negative predicted values (backgrounds): ', neg_ious, '/', f'{neg_ious / all_ious: .2%}')
print('Total predicted values: ', all_ious)

tp = pos_ious
fp = neg_ious
fn = 498 - pos_ious
recall = tp / (tp + fn)
print('Recall: ', f'{recall:.2%}')

precision = tp / (tp + fp)
print('Precision: ', f'{precision:.2%}')

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
precisions, recalls = compute_precision_recall(test_re, thresholds)
plot_map(precisions, recalls)