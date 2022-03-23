def yolo(df, path_cfg, path_weights, w = 416, h = 416, threshold = 0.5):

  '''

  This function implements yolo pre-trained model.

  Requirements: downloaded .cfg and .weights files. To obtain these, plese see: https://pjreddie.com/darknet/yolo/

  Params: 
  - df with at least the following attributes:
    0: image path

  - patch_cfg: path with the configuration file from yolo pretrained model
  - path_weights: path with the weights from yolo pretrained model
  

  - w: width of resized image. Default 416
  - h: height of resized image. Default 416

  '''
  
  boxes_all = []
  confidence_all = []
  class_id_all = []

  for i in df.iloc:
    img = cv2.imread(str(i[0]))
    height, width, shape = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (w,h), (0,0,0), True, crop = False) # extract features. Normalize and resize. Swap RGB colours
    
    net = cv2.dnn.readNet(path_cfg, path_weights)
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    net.setInput(blob)
    outs = net.forward(outputlayers)

    
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
      for detection in out:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        confidences.append(float(confidence))
        class_ids.append(class_id)
        if confidence > threshold: ##### set up threshold
          # object detection
          centre_x = int(detection[0])*width
          centre_y = int(detection[1])*height
          w = int(detection[2])*width
          h = int(detection[3])*height

          # rectangle coordinates
          x = int(centre_x - w/2)
          y = int(centre_y - h/2)

          boxes.append([x,y,w,h])
    
    boxes_all.append(boxes)
    confidence_all.append(confidences)
    class_id_all.append(class_ids)

  # add values to df
  df['boxes'] = boxes_all
  df['confidence'] = confidence_all
  df['class_id'] = class_id_all

  return df

