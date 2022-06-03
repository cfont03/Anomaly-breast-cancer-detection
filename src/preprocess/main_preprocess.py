import pandas as pd
import numpy as np
from PIL import Image

if __name__ == '__main__':
  from utils.generate_mask import generate_mask
  from utils.horizontal_flip import horizontal_flip
  from utils.map_img_txt import map_img_txt
  from utils.noise import remove_noise
  from utils.rotation import rotate_images
  from utils.train_test_split import train_test_split
  from utils.vertical_flip import vertical_flip
  from utils.txt_csv import txt_to_csv
else:
    pass

df_info = pd.read_csv("res/Info.txt", sep = " ", header = 'infer')
df0 = txt_to_csv(path_in="res/Info.txt", path_out="outputs/images_info_0.csv")
df1 = map_img_txt(path_in='outputs/images_info_0.csv', path_out='outputs/images_info_1.csv')
train, test = train_test_split(df1)


### Data Augmentation Techniques
x_y_hf_train, x_min_max_hf_train, y_min_max_hf_train = horizontal_flip(train)
x_y_hf_test, x_min_max_hf_test, y_min_max_hf_test = horizontal_flip(test)

remove_noise(train)
remove_noise(test)

x_y_train, x_min_max_train, y_min_max_train = rotate_images(train)
x_y_test, x_min_max_test, y_min_max_test = rotate_images(test)

generate_mask(train)
generate_mask(test)

x_y_vf_train, x_min_max_vf_train, y_min_max_vf_train = vertical_flip(train)
x_y_vf_test, x_min_max_vf_test, y_min_max_vf_test = vertical_flip(test)


### Map training dataset
data = []
x_coord = []
y_coord = []
categ = []
xmin = []
ymin = []
xmax = []
ymax = []
size = []
names = []
radius = []
image_path = []
masks = ['_mask']
hflips = ['_hflip']
noises = ['_noise']
vflips = ['_vflip']
rotate = ['_rotate']

with open('outputs/train.csv') as file:
  next(file) # skip header
  z = 0
  for l in file:
    for r in rotate:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.jpeg'.format(i, r)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, r)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(x_y_train[z][0][0])
      y_coord.append(x_y_train[z][0][1])
      xmin.append(x_min_max_train[z][0][0])
      xmax.append(x_min_max_train[z][0][1])
      ymin.append(y_min_max_train[z][0][0])
      ymax.append(y_min_max_train[z][0][1])
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)

    for h in hflips:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.jpeg'.format(i, h)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, h)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(x_y_hf_train[z][0][0])
      y_coord.append(x_y_hf_train[z][0][1])
      xmin.append(x_min_max_hf_train[z][0][0])
      xmax.append(x_min_max_hf_train[z][0][1])
      ymin.append(y_min_max_hf_train[z][0][0])
      ymax.append(y_min_max_hf_train[z][0][1])
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)

    for v in vflips:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.jpeg'.format(i, v)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, v)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(x_y_vf_train[z][0][0])
      y_coord.append(x_y_vf_train[z][0][1])
      xmin.append(x_min_max_vf_train[z][0][0])
      xmax.append(x_min_max_vf_train[z][0][1])
      ymin.append(y_min_max_vf_train[z][0][0])
      ymax.append(y_min_max_vf_train[z][0][1])   
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)

    for n in noises:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.jpeg'.format(i, n)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, n)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(np.array(l.split(",")[6]))
      y_coord.append(np.array(l.split(",")[7]))
      xmin.append(np.array(l.split(",")[10]))
      xmax.append(np.array(l.split(",")[11]))
      ymin.append(np.array(l.split(",")[12]))
      ymax.append(int(np.array(l.split(",")[13])))
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)

    for m in masks:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.pgm'.format(i, m)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, m)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(np.array(l.split(",")[6]))
      y_coord.append(np.array(l.split(",")[7]))
      xmin.append(np.array(l.split(",")[10]))
      xmax.append(np.array(l.split(",")[11]))
      ymin.append(np.array(l.split(",")[12]))
      ymax.append(int(np.array(l.split(",")[13])))
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)
      
    z +=1

arr = np.array([names, data, size, categ, x_coord, y_coord, xmin, xmax, ymin, ymax, radius, image_path], dtype = object).T.tolist()
train_ = pd.DataFrame(data = arr, columns = ['name', 'features', 'size', 'cat', 'x_coord', 'y_coord', 'xmin', 'xmax', 
                                          'ymin', 'ymax', 'radius', 'image path'])
train_pp = pd.concat([train, train_])
train_pp['radius'].apply(float).apply(int) + train_pp['x_coord'].apply(float).apply(int) + train_pp['y_coord'].apply(float).apply(int) 
+ train_pp['xmin'].apply(int)+ train_pp['xmax'].apply(int) + train_pp['ymin'].apply(int) + train_pp['ymax'].apply(int)
train_pp.to_csv("outputs/images_preprocess_train.csv", sep = ";")


### Map testing dataset
data = []
x_coord = []
y_coord = []
categ = []
xmin = []
ymin = []
xmax = []
ymax = []
size = []
names = []
radius = []
image_path = []
masks = ['_mask']
hflips = ['_hflip']
noises = ['_noise']
vflips = ['_vflip']
rotate = ['_rotate']

with open('outputs/test.csv') as file:
  next(file) # skip header
  z = 0
  for l in file:
    for r in rotate:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.jpeg'.format(i, r)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, r)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(x_y_test[z][0][0])
      y_coord.append(x_y_test[z][0][1])
      xmin.append(x_min_max_test[z][0][0])
      xmax.append(x_min_max_test[z][0][1])
      ymin.append(y_min_max_test[z][0][0])
      ymax.append(y_min_max_test[z][0][1])
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)

    for h in hflips:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.jpeg'.format(i, h)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, h)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(x_y_hf_test[z][0][0])
      y_coord.append(x_y_hf_test[z][0][1])
      xmin.append(x_min_max_hf_test[z][0][0])
      xmax.append(x_min_max_hf_test[z][0][1])
      ymin.append(y_min_max_hf_test[z][0][0])
      ymax.append(y_min_max_hf_test[z][0][1])
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)

    for v in vflips:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.jpeg'.format(i, v)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, v)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(x_y_vf_test[z][0][0])
      y_coord.append(x_y_vf_test[z][0][1])
      xmin.append(x_min_max_vf_test[z][0][0])
      xmax.append(x_min_max_vf_test[z][0][1])
      ymin.append(y_min_max_vf_test[z][0][0])
      ymax.append(y_min_max_vf_test[z][0][1])   
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)

    for n in noises:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.jpeg'.format(i, n)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, n)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(np.array(l.split(",")[6]))
      y_coord.append(np.array(l.split(",")[7]))
      xmin.append(np.array(l.split(",")[10]))
      xmax.append(np.array(l.split(",")[11]))
      ymin.append(np.array(l.split(",")[12]))
      ymax.append(int(np.array(l.split(",")[13])))
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)

    for m in masks:
      i = l.split(",")[1]
      path = 'res/all-mias/{:}{:}.pgm'.format(i, m)
      img = Image.open(path)
      data_arr = np.array(img.getdata())
      width, height = np.array(img.size)
      data.append(data_arr)
      size.append((width, height))
      names.append(np.array('{:}{:}'.format(i, m)))
      categ.append(np.array(l.split(",")[5]))
      x_coord.append(np.array(l.split(",")[6]))
      y_coord.append(np.array(l.split(",")[7]))
      xmin.append(np.array(l.split(",")[10]))
      xmax.append(np.array(l.split(",")[11]))
      ymin.append(np.array(l.split(",")[12]))
      ymax.append(int(np.array(l.split(",")[13])))
      radius.append(np.array((l.split(",")[8]).rstrip()))
      image_path.append(path)
      
    z +=1

arr = np.array([names, data, size, categ, x_coord, y_coord, xmin, xmax, ymin, ymax, radius, image_path], dtype = object).T.tolist()
test_ = pd.DataFrame(data = arr, columns = ['name', 'features', 'size', 'cat', 'x_coord', 'y_coord', 'xmin', 'xmax', 
                                          'ymin', 'ymax', 'radius', 'image path'])
test_pp = pd.concat([test, test_])
test_pp['radius'].apply(float).apply(int) + test_pp['x_coord'].apply(float).apply(int) + test_pp['y_coord'].apply(float).apply(int) 
+ test_pp['xmin'].apply(int)+ test_pp['xmax'].apply(int) + test_pp['ymin'].apply(int) + test_pp['ymax'].apply(int)
test_pp.to_csv("outputs/images_preprocess_test.csv", sep = ";")