import numpy as np
import cv2
import matplotlib.pylab as plt
from pathlib import Path


def anchor_points (df, w = 800, h = 800):

    '''
    
    This function plots an indicated image with its possible anchor boxes and adds the centre of coordinates x and y into columns in df.


    Params: df with:
    1: image path
    2: min x coord
    3: max x coord
    4: min y coord
    5: max y coord
    6: class
    7: number of anchors
    8: feature map width
    9: feature map height

    width and height of df default at 800

    Output: plot of random image with possible anchors and df with additional info:
    - Number of possible anchors per image (n_anchors)
    - width of feature map (w_fm)
    - height of feature map (h_fm)
    - centre_list: list with (x,y) anchor points of the image
    
    '''


    # -------- calculate the centre of anchor image for all images --------

    centre_lists, x_strs, y_strs = [], [], []

    for i in df.iloc:
      img_path = i[1]
      
      x = df[df['image path'].apply(lambda x : x == str(Path(img_path)))].iloc[:,8]
      
      x_str = int(w / int(x.values[0]))
      y = df[df['image path'].apply(lambda x : x == str(Path(img_path)))].iloc[:,9]
      y_str = int(h / int(y.values[0]))
      
      x_centre = np.arange(8, w, x_str) 
      y_centre = np.arange(8, h, y_str)
      centre_list = np.array(np.meshgrid(x_centre, y_centre,  sparse=False, indexing='xy')).T.reshape(-1,2)

      centre_lists.append(centre_list)

    df['x_str'] = x_str
    df['y_str'] = y_str
    df['anchor_points'] = centre_lists



    # ------------- plot image -------------

    # take indicated image
    image = input("introduce image name: ") 
    img_path = 'res/all-mias/{:}'.format(image)
    img_ = cv2.imread(img_path)
    
    centre_list_r1 = df[df['image path'].apply(lambda x : x == str(Path(img_path)))].iloc[:,-1]
    centre_list_r1 = centre_list_r1.values[0]

    # plot anchor positions 
    plt.figure(figsize=(9, 6))
    

    a = df[df['image path'].apply(lambda x : x == str(Path(img_path)))].iloc[:,7]
    b = a.values[0]
    for i in range(b):
      cv2.circle(img_, (int(centre_list_r1[i][0]), int(centre_list_r1[i][1])), radius=1, color=(255, 0, 0), thickness=5) 
      
    
    plt.imshow(img_)
    plt.show()


    return df