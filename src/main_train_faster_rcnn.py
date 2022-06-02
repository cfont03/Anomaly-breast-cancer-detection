import pandas as pd
import numpy as np
import time
import ssl
import keras

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    from preprocess.utils.resize_inputs import resize_input
    from train.utils.anchor_boxes import anchor_boxes
    from train.utils.anchor_points import anchor_points
    from train.utils.sample_anchors_pre import sample_anchors_pre
    from train.utils.precision_recall import compute_precision_recall
    from train.utils.non_max_supression import non_max_suppression_fast
    from train.utils.plots import boundary_box, plot_loss_rpn, plot_map
    from train.utils.intersection_over_union import IoU_faster_rcnn, IoU_ROI
    from train.utils.coordinate_labels import coord_labels
    from train.algorithms.faster_rcnn.regional_proposal_network import rpn
    from train.algorithms.faster_rcnn.regions_of_interest import rpn_to_roi
    from train.algorithms.faster_rcnn.vgg16_cnn import vgg16_cnn
else:
    pass

train = pd.read_csv('outputs/images_preprocess_train.csv', sep = ";")

# plot selected image
boundary_box(train)

train_ = train.iloc[:, [8, 9, 10, 11, 12]]
train_['height'] = 1024
train_['weight'] = 1024
train_['cat'] = train['cat']


### Adapt to fit VGG16
# resize to 800 x 800
train_re = resize_input(train_, w = 800, h = 800)

# calculate feature map
train_re, vgg16_model, feature_map = vgg16_cnn(train_re)


### CHECKPOINT
  ## Save files
train_re.to_csv("outputs/train_re.csv", sep = ";")
vgg16_model.save('outputs/vgg16_model.h5')
np.save('outputs/feature_map', feature_map)
  ## Upload files
vgg16_model = keras.models.load_model('outputs/vgg16_model.h5', compile = False)
train_re = pd.read_csv('outputs/train_re.csv', sep = ";")
feature_map = np.load('outputs/feature_map.npy')


### Anchor points and boxes
anchor_points(train_re)
anchor_boxes(train_re)
IoU_faster_rcnn(train_re)
sample_anchors_pre(train_re)
offset_list, label_list = coord_labels(train_re)

offset_list_label_list = np.zeros((498, 22500, 5))

layer1 = 0
for i in offset_list:
  for z in label_list:
    offset_list_label_list[layer1, :] = np.column_stack((i, z))[np.newaxis,:]
  
  layer1 +=1

### Regional Proposal Network
rpn_model, x_deltas, x_scores = rpn(vgg16_model)

start_time = time.time()
rpn_hist = rpn_model.fit(feature_map,[offset_list_label_list, label_list], epochs= 100)
print(f"\n------- Execution time: {(time.time() - start_time)/60:.2f} minutes -------\n")  
 
    ### CHECKPOINT
rpn_model.save('outputs/rpn_model_vgg16.h5')

plot_loss_rpn(rpn_hist)


### Predictions
pred_x_deltas, pred_x_scores = rpn_model.predict(feature_map)

n_anchors = 22500
pred_x_deltas = pred_x_deltas.reshape(-1, n_anchors, 4) 
pred_x_scores = pred_x_scores.reshape(-1, n_anchors)

dx = pred_x_deltas[:, :, 0]
dy = pred_x_deltas[:, :, 1]
dw = pred_x_deltas[:, :, 2]
dh = pred_x_deltas[:, :, 3]

  ### CHECKPOINT
np.save('outputs/pred_x_scores', pred_x_scores)
np.save('outputs/pred_x_deltas', pred_x_deltas)  


### Regions of Interest
roi, scores_sorted, rois_sorted, sorted_idxs = rpn_to_roi(train_re, pred_x_scores, dx, dy, dw, dh)
roi_sorted_filtered, score_sorted_filtered = non_max_suppression_fast(rois_sorted, scores_sorted)
roi_sorted_filter = np.array(roi_sorted_filtered)
score_sorted_filter = np.array(score_sorted_filtered)

train_re['roi_filter'] = roi_sorted_filter
train_re['score_filter'] = score_sorted_filter

train_re, idx_pos, idx_neg = IoU_ROI(train_re)


### Model evaluation
rois_pos = []
for i, z in zip(train_re['roi_filter'], train_re['iou_idx_pos']):
  roi_pos = i[z]
  rois_pos.append(roi_pos)

train_re['rois_pos'] = rois_pos

# count positive and negative ious
pos_ious = 0
for i in train_re.iloc[:,-1]:
  if(len(i) > 0):
    pos_ious += 1
  else:
    pos_ious += 0

neg_ious = 0
for i in train_re.iloc[:,-2]:
  if(len(i) > 0):
    neg_ious += 1
  else:
    neg_ious += 0

all_ious = 0
for i in train_re.iloc[:,20]:
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
precisions, recalls = compute_precision_recall(train_re, thresholds)
plot_map(precisions, recalls)