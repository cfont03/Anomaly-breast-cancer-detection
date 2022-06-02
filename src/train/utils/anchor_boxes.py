import cv2
import numpy as np
import matplotlib.pylab as plt
from pathlib import Path


def anchor_boxes (df, w= 800, h = 800):

    
  ''' This function calculates the anchor boxes per image and plots one random image. It additionally adds the anchor boxes in an array.
  
  Params: 
  
  - df with:
  1: image path
  2: min x coord
  3: max x coord
  4: min y coord
  5: max y coord
  6: class
  7: number of anchors
  8: feature map of width
  9: feature map height
  10: anchor centre of fm, x coord
  11: anchor centre of fm, y coord
  12: list of centre coords in img (x,y)

  width and height. default at 800

  Output: plot of random image with 9 anchor boxes at the middle of the centre anchor and df with additional info:
  - possible anchor boxes per image with xmin, ymin, xmax, ymax
  - possible anchor boxes per image with x, y, w, h
    
  '''

    
  # ------------- calculate anchor boxes per image -------------

  l = []
        
  # aspect ratio width and height
  anchor_ratio_l = [0.5, 1, 2] # 3 scenarios: vertical rect (width is half height), square (width = height), horizontal rect (width is twice hight)
  anchor_scale_l = [8, 16, 32] # anchor box area, according to original paper [128, 256, 512] in original paper


  total_anchors = []
  n_objects = []
  anchors_list, anchors_lists = [], []
  anchors_list_cent, anchors_lists_cent = [], []

  # rpn is passed into conv. layer with 3x3 filters
  for i in df.iloc:
    n_anchors = i[7] * len(anchor_ratio_l) * len(anchor_scale_l) # number of possible anchors
    total_anchors.append(n_anchors)
    bbox_list = np.array([i[2], i[4], i[3], i[5]])
    n_object = len(bbox_list)
    n_objects.append(n_object)
    # there are a total of 2500 anchor centres per image, each anchor centre has 9 anchor boxes
    # total number of anchor boxes in feature map will be 2500 * 9 = 22500, and each anchor box is denoted by 4 numbers
    anchor_list = np.zeros(shape= (n_anchors, 4))
    anchor_list_cent = np.zeros(shape= (n_anchors, 4))
          
    count = 0
    centre_list_ = i[-1]
    x_str = i[-3]
    y_str = i[-2]


    for centre in centre_list_:
      centre_x, centre_y = centre[0], centre[1]
      for ratio in anchor_ratio_l:
        for scale in anchor_scale_l:
          h = pow(pow(scale, 2)/ ratio, 0.5)
          w = h * ratio

          h *= x_str
          w *= y_str


          anchor_xmin = centre_x - 0.5 * w
          anchor_xmax = centre_x + 0.5 * w
          anchor_ymin = centre_y - 0.5 * h
          anchor_ymax = centre_y + 0.5 * h
          l.append([centre_x, centre_y, w, h])     
          
          anchor_list[count] = [anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax]
          anchor_list_cent[count] = [centre_x, centre_y, w, h]
                
          count += 1

    anchors_lists.append(anchor_list)      
    anchors_lists_cent.append(anchor_list_cent)
      
  # add values to the df
  df['n_possible_anchors'] = total_anchors
  df['n_object'] = n_objects
  df['possible_anchors'] = anchors_lists
  df['possible_anchors_cent'] = anchors_lists_cent


  
  # ------------ visualize anchor boxes at center anchor location of an indicated image

  # take indicated image
  image = input("introduce image name: ") 
  img_path = 'res/all-mias/{:}'.format(image)
  img_ = cv2.imread(img_path)

  centre_list_r2 = df[df['image path'].apply(lambda x : x == str(Path(img_path)))].iloc[:,12]
  centre_list_r2 = centre_list_r2.values[0]


  # mid anchor centre = 2500/2 = 1250
  anchor = df[df['image path'].apply(lambda x : x == str(Path(img_path)))].iloc[:,7]
  anchor = anchor.values[0]
  range1 = int(round(anchor/2 * 9, 0))


  for i in range(11025, 11025+9):  # 1250 * 9 = 11025 (9 anchors corresponds to mid anchor center)
    x_min = int(anchor_list[i][0])
    y_min = int(anchor_list[i][1])
    x_max = int(anchor_list[i][2])
    y_max = int(anchor_list[i][3])
    cv2.rectangle(img_, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=3) 

  bbox_lists = []
  for i in df.iloc:
    bbox_list = np.array([i[2], i[4], i[3], i[5]])
    bbox_lists.append(bbox_list)

  df['bbox_list'] = bbox_lists


  ground_truth = df[df['image path'].apply(lambda x : x == str(Path(img_path)))].iloc[:,-1]
  ground_truth = ground_truth.values[0]

  cv2.rectangle(img_, (int(ground_truth[0]), int(ground_truth[1])), (int(ground_truth[2]), int(ground_truth[3])), color=(0, 0, 255), thickness=3)       
  
  cv2.circle(img_, (int(centre_list_r2[312][0]), int(centre_list_r2[312][1])), radius=1, color=(0, 255, 0), thickness=15) 

  plt.imshow(img_)
  plt.show()


  return df