import PIL.Image
import numpy as np


data = []
x_coord = []
y_coord = []
size = []
names = []
radius = []
image_path = []
cat = []

path_labels = '/content/archive/images_info_0.csv'

with open(path_labels) as file:
  next(file)
  for l in file:
    i = l.split(",")[1]
    images_path = '/content/archive/all-mias/{:}.pgm'.format(i)
    img = PIL.Image.open(images_path)
    data_arr = np.array(img.getdata())
    width, height = np.array(img.size)
    data.append(data_arr)
    size.append((width, height))
    names.append(np.array(i))
    cat.append(np.array(l.split(",")[2]))
    x_coord.append(np.array(l.split(",")[3]))
    y_coord.append(np.array(l.split(",")[4]))
    radius.append(np.array((l.split(",")[5]).rstrip()))
    image_path.append(images_path)