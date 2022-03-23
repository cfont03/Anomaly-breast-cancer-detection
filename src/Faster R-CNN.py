# import of libraries
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from urllib import request
from tensorflow.keras import initializers
np.set_printoptions(suppress=True)
from pathlib import Path
from numpy.core.fromnumeric import shape



#### ----------- resizing input images to fit into pretrained model to extract features
def resize (df, h = 800, w = 800):
  
  '''

  Function returns resized images to specified size. Format: df
  Default is 800 x 800.

  Params:
  0: image path
  1: xmin
  2: xmax
  3: ymin
  4: ymax
  5: height original image
  6: weight original image
  7: label

  '''

  paths = []
  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  labels = []

  for i in df.iloc:
    img = cv2.imread(str(i[0]))
    img = cv2.resize(img, dsize = (h,w), interpolation = cv2.INTER_CUBIC)
    path, filename = os.path.split(i[0])
    new_path = Path(str(path)+'/{:}_resize{:}'.format(i[0][-10:-4], '.jpeg')) # generate new path
    status = cv2.imwrite(str(new_path), img)
    print("Image written to file-system " , new_path,  " :", status) # check if saved
    paths.append(new_path)

    x_factor = w / i[5]
    y_factor = h / i[6]

    # adapt bounding box accordingly
    xmin = i[1] * x_factor
    xmins.append(xmin)
    xmax = i[2] * x_factor
    xmaxs.append(xmax)
    ymin = i[3] * y_factor
    ymins.append(ymin)
    ymax = i[4] * y_factor
    ymaxs.append(ymax)

    labels.append(i[7])

  arr = np.array([paths, xmins, xmaxs, ymins, ymaxs, labels], dtype = object).T.tolist()
  df_ = pd.DataFrame(data = arr, columns = ['path', 'xmin', 'xmax', 'ymin', 'ymax', 'labels'])
  
  return df_
  
  
  
## ---------------- show boundary box to resized image

def boundary_box_resize (df):
  
  '''

  This function plots an image with its boundary boxes

  Params:
  0: path image
  1: min coord x
  2: max coord x
  3: min coord y
  4: max coord y

  '''

  i = df.iloc[0]
  image = input("introduce image name: ")
  path, filename = os.path.split(i[0])
  im = cv2.imread(str(path) + '/{:}_resize.jpeg'.format(image))

  # generate x, y, width and height
  xmin, xmax, ymin, ymax = (i[1], i[2], i[3], i[4])
  width = xmax - xmin
  height = ymax - ymin

  # Create figure and axes
  fig, ax = plt.subplots()

  # Display the image
  ax.imshow(im)

  # Create a Rectangle patch
  rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')

  # Add the patch to the Axes
  ax.add_patch(rect)

  plt.show()
  
  
  
  

## ---------------- VGG16 Network

def vgg16_cnn(df, h = 800, w = 800):

  '''

  Returns a VGG16 model and a list of anchors needed per image passed

  Params: df with
  0: image path

  '''


  vgg = tf.keras.applications.VGG16(
    include_top=False, weights='imagenet')

  input = vgg.input
  output = vgg.layers[-1].output

  vgg16_model = Model(inputs = input, outputs = output)
  
  
  # feature map
  anchors = []
  w_fms = []
  h_fms = []

  for i in df.iloc:
    img = cv2.imread(str(i[0]))
    shape_fm = vgg16_model.predict(np.expand_dims(img, 0))
    _, w_fm, h_fm, _ = shape_fm.shape
    n_anchor = w_fm * h_fm    # calculate number of anchors per image
    anchors.append(n_anchor)
    w_fms.append(w_fm)
    h_fms.append(h_fm)


  # add feature map in the df
  df['n_anchor'] = anchors
  df['w_fm'] = w_fms
  df['h_fm'] = h_fms


  return print('Number of anchors needed: ', n_anchor), vgg16_model.summary()
  


  
## --------- calculate anchors

def anchors(df, w = 800, h = 800, anchor = 625):

  '''

  This function plots a random image with its anchors. Additionally it calculates anchor boxes per images 
  and ultimatley plots them.

  Params: df with:
  0: image path
  1: min x coord
  2: max x coord
  3: min y coord
  4: max y coord
  5: class
  6: number of anchors
  7: feature map of width
  8: feature map height

  ''' 
    
  def plot_anchors(df):

    '''
    
    This function plots a random image with its anchors
    
    '''

    # take one random image
    path, filename = os.path.split(df.iloc[0][0]) # take path from first element in the df
    img_rand = random.choice([x for x in os.listdir(path) 
                            if (os.path.isfile(os.path.join(path, x)) and len(x) >= 18)]) # only image with '_resize'
    
    img_path = str(path)+'/{:}'.format(img_rand)
    img_ = cv2.imread(img_path)
    x_str = int(w / int(df[df['path'].apply(lambda x : x == Path(img_path))].iloc[:,-2]))
    y_str = int(h / int(df[df['path'].apply(lambda x : x == Path(img_path))].iloc[:,-1]))
    #print(x_str)
    # centre anchor image
    x_centre = np.arange(8, w, x_str) 
    y_centre = np.arange(8, h, y_str) 
    
    centre_list = np.array(np.meshgrid(x_centre, y_centre,  sparse=False, indexing='xy')).T.reshape(-1,2)

    # plot anchor positions 
    plt.figure(figsize=(9, 6))
    
    #for i in range(df['path'].isin([img_][5])):
    for i in range(625):
      cv2.circle(img_, (int(centre_list[i][0]), int(centre_list[i][1])), radius=1, color=(255, 0, 0), thickness=5) 
      
    print('Image name: ', img_rand)
    plt.imshow(img_)
    plt.show()



    def anchor_boxes(df, centre_list, x_str, y_str, img_):

      ''' This function calculates the anchor boxes per image '''

      l = []
        
      # aspect ratio width and height
      anchor_ratio_l = [0.5, 1, 2] # 3 scenarios: vertical rect (width is half height), square (width = height), horizontal rect (width is twice hight)
      anchor_scale_l = [8, 16, 32] # anchor box area


      total_anchors = []
      n_objects = []
      anchors_list = []
      anchors_lists = []

      for i in df.iloc:
        n_anchors = i[6] * len(anchor_ratio_l) * len(anchor_scale_l) # number of possible anchors
        total_anchors.append(n_anchors)
        bbox_list = np.array([i[1], i[3], i[2], i[4]]) # boundary box in np
        n_object = len(bbox_list) # number of objects in the image
        n_objects.append(n_object)
        anchor_list = np.zeros(shape= (n_anchors, 4))
          
      
        count = 0
        # for each anchor
        for centre in centre_list:
            centre_x, centre_y = centre[0], centre[1]
            # for each ratio
            for ratio in anchor_ratio_l:
              # for each scale
              for scale in anchor_scale_l:
                # compute height and width and scale them by constant factor
                h = pow(pow(scale, 2)/ ratio, 0.5)
                w = h * ratio

                # scale h and w
                h *= x_str
                w *= y_str


                # * at this point we have height and width of anchor and centers of anchor locations
                # putting anchor 9 boxes at each anchor locations
                anchor_xmin = centre_x - 0.5 * w
                anchor_xmax = centre_x + 0.5 * w
                anchor_ymin = centre_y - 0.5 * h
                anchor_ymax = centre_y + 0.5 * h
                l.append([centre_x, centre_y, w, h])
                      
                # append the anchor box to anchor list
                anchor_list[count] = [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax]
                anchors_list.append(anchor_list)
                
                count += 1
        
        anchors_lists.append(anchor_list)
      
      # add values to the df

      df['possible_anchors'] = total_anchors
      df['n_object'] = n_objects
      df['anchor_list'] = anchors_lists


      # visualize anchor boxes at center anchor location
      img_ = np.copy(img_)
      # mid anchor center = 625/2 = 312.5
      range1 = int(round(anchor/2 * 9, 0))
      for i in range(range1, range1+9):  # 312.5 * 9 = 2812.5 (9 anchors corresponds to mid anchor center)
          x_min = int(anchor_list[i][0])
          y_min = int(anchor_list[i][1])
          x_max = int(anchor_list[i][2])
          y_max = int(anchor_list[i][3])
          cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 

      bbox_lists = []
      for i in df.iloc:
        bbox_list = np.array([i[1], i[3], i[2], i[4]])
        bbox_lists.append(bbox_list)

      df['bbox_list'] = bbox_lists


      for i, bbox in enumerate(df_re_red.iloc[:,-1]):
        cv2.rectangle(img_, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0), thickness=3)
          
      cv2.circle(img_, (int(centre_list[312][0]), int(centre_list[312][1])), radius=1, color=(0, 0, 255), thickness=15) 

      plt.imshow(img_)
      plt.show()


    return anchor_boxes(df, centre_list, x_str, y_str, img_)

  return plot_anchors(df)  




### --------------- calculate IoU

def IOU(df):
    
    '''

    Calculate Intersection of Union (IoU) between predicted box and ground truth.

    Params: df containing info on images
    0: image path
    1: x min coord
    2: x max coord
    3: y min coord
    4: y max coord
    5: label
    6: number of anchors
    7: width feature map
    8: height feature map
    9: number of possible anchors
    10: number of objects
    11: anchor: predicted box in array. Sorted by xmax, ymax, xmin, ymin
    12: boundary box: ground truth in array. Sorted by xmax, ymax, xmin, ymin
    
    '''
    

    def boxes_in_image(df, h = 800, w = 800):

      ''' 

      Function returns the anchor boxes which are indeed inside the image

      '''

      anchors = df.iloc[:,-2]
      in_anchor_lists = []
      n_in_anchors = []

      for i in anchors:
        in_anchor_idx_list = np.where(
            (i[:,0] >= 0) &
            (i[:,1] >= 0) &
            (i[:,2] <= w) & 
            (i[:,3] <= h))[0]
        in_anchor_list = i[in_anchor_idx_list]
        in_anchor_lists.append(in_anchor_list)
        n_in_anchor = len(in_anchor_idx_list)
        n_in_anchors.append(n_in_anchor)

      df['inside_anchor_list'] = inside_anchor_lists
      df['n_inside_anchor'] = n_inside_anchors

      return df


    # ------calculate coordinate of overlapping region------

    area_overlaps = []
    ious = []

    truth = df.iloc[:,-3]
    pred = df.iloc[:,-2]
    
    for truth, pred in zip(truth, pred):
      # take max of x1 and y1 out of both boxes
      x1 = max(truth[0], pred[0][0])
      y1 = max(truth[1], pred[0][1])
      # take min of x2 and y2 out of both boxes
      x2 = min(truth[2], pred[0][2])
      y2 = min(truth[3], pred[0][3])
    
      
      # ------area of overlapping region------
      width_overlap = (x2 - x1)
      height_overlap = (y2 - y1)
      area_overlap = width_overlap * height_overlap
      area_overlap = [x for x in area_overlap if x > 0]
      area_overlaps.append(area_overlap)      

    
      # ------computing union------
      # sum of area of both the boxes - area_overlap
      
      # height and width of both boxes
      width_truth = (truth[2] - truth[0])
      height_truth = (truth[3] - truth[1])
      
      width_pred = (pred[0][2] - pred[0][0])
      height_pred = (pred[0][3] - pred[0][1])
      
      # area of box1 and box2
      area_truth = width_truth * height_truth
      area_pred = width_pred * height_pred
      
      # union (including 2 * overlap area (double count))
      area_union_overlap = area_truth + area_pred
      
      # union
      area_union = area_union_overlap - area_overlap
      
      # compute IOU
      iou = area_overlap/ area_union
      ious.append(iou)


    df['area_overlap'] = area_overlaps
    df['iou'] = ious
    
    return df

