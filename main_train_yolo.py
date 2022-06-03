import pandas as pd
import numpy as np
import time
import cv2
from keras.optimizers import Adam
import keras.backend as K


if __name__ == "__main__":
    from preprocess.utils.resize_inputs import resize_input
    from train.utils.precision_recall import compute_precision_recall_yolo
    from train.utils.plots import boundary_box, plot_loss_yolo, plot_map
    from train.utils.intersection_over_union import IoU_yolo
    from train.algorithms.yolo.cell_to_img import cell_to_img
    from train.algorithms.yolo.loss_function import yolo_sse
    from train.algorithms.yolo.yolo_grid import yolo_grid
    from train.utils.boundary_box import boundary_box_yolo
    from train.utils.coordinate_cells import coord_cells
else:
    pass


train = pd.read_csv('outputs/images_preprocess_train.csv', sep = ";")

# plot selected image
boundary_box(train)

train_ = train.iloc[:, [8, 9, 10, 11, 12]]
train_['height'] = 1024
train_['weight'] = 1024
train_['cat'] = train['cat']


### Adapt to fit CNN
# resize to 448 x 448
train_re = resize_input(train_, alg = 'yolo', h = 448, w = 448)
train_re1 = boundary_box_yolo(train_re, 7, 1, 1)
img_list, cell_list = coord_cells(train_re1)

### CNN
model_yolo = yolo_grid(3) # 3 for RGB

train_images = []

for i in train_re.iloc:
    img = cv2.imread(str(i[0]))
    train_images.append(img)

train_images = np.array(train_images)
train_images = train_images/255 # normalize images

optimizer = Adam(learning_rate = 1e-5)
model_yolo.compile(loss=yolo_sse, optimizer=optimizer, run_eagerly=True)
start_time = time.time()
model_hist = model_yolo.fit([train_images, cell_list], cell_list, epochs = 50)
print(f"\n------- Execution time: {(time.time() - start_time)/60:.2f} minutes -------\n")

    ## CHECKPOINT
model_yolo.save('outputs/model_yolo.h5')

plot_loss_yolo(model_hist)

### Prediction
y_pred = model_yolo.predict([train_images, cell_list])
S = 7
B = C = 1
y_pred = y_pred.reshape(-1, S, S, B * 5 + C)

# convert predictex boxes relative to img
y_pred_img = cell_to_img(y_pred, B = 1, C = 1)

img_list_ = K.reshape(img_list, (-1, 1, 5))
y_pred_img_ = K.reshape(y_pred_img, (-1, 49, 5))

ious = np.zeros((498 * 49))
layer1 = 0

for grid_t, grid_pred in zip(img_list_, y_pred_img_):
  for t in grid_t: 
    for p in grid_pred:
      iou = IoU_yolo(t, p)
      ious[layer1] = iou
      
      layer1 += 1

ious = ious.reshape(498, 49, 1)


### Model evaluation
scores = np.zeros((498, 7 * 7, 1))

threshold = 0.5

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


print('TP: Positive predicted values (foregrounds): ', len(scores[scores == 2]), '/', f'{len(scores[scores == 2]) / len(scores.reshape(-1)): .2%}')
print('FP: Negative predicted values (backgrounds): ', len(scores[scores == 1]), '/', f'{len(scores[scores == 1]) / len(scores.reshape(-1)): .2%}')
print('Total predicted values: ', len(scores.reshape(-1)))

print('Correctly predicted boxes out of total predictions: ', f'{len(scores[scores == 2]) / len(scores.reshape(-1)):.2%}')

tp = len(scores[scores == 2])
fp = len(scores[scores == 1])
fn = len(scores[scores == 0])

recall = tp / (tp + fn)
print('Recall: ', f'{recall:.2%}')

precision = tp / (tp + fp)
print('Precision: ', f'{precision:.2%}')

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
precisions, recalls = compute_precision_recall_yolo(ious, thresholds)
plot_map(precisions, recalls)