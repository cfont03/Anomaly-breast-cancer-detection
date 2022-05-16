import cv2


def remove_noise (info_file):

  '''
  
  Function removes noise of images contained in a file
  
  Args: df with Params:
    0: reference number 
    1: features
    2: size
    3: class
    4: x_coordinate of the abnormality
    5: y_coordiante of the abnormality
    6: radius in pixels
    7: path of the image

  Gaussian noise is assumed

  Output: saved images without noise
  
  '''

  for name, path in zip(info_file.iloc[:, 0], info_file.iloc[:, 7]):
    noise_ = cv2.imread(path)
    noise = cv2.fastNlMeansDenoising(noise_,None,10,7,21)
    new_path = '/content/archive/all-mias/{:}_noise.jpeg'.format(name)
    status = cv2.imwrite(str(new_path), noise)

    ### CHECKPOINT OUTPUT
    print("Image written to file-system " , new_path,  " :", status)