def train_test_split(df, test = 0.3):

  '''

  This function splits data into train and test subsets.


  Params:

  dataframe:
    0: image name
    1: features
    2: size image
    3: class
    4: x centre coord of boundary box
    5: y centre coord of boundary box
    6: radius
    7: image path
    9, 10, 11, 12: coord of boundary box in xmax, xmin, ymax, ymin order
  
  test: % of test data of all dataframe


  '''

  ### CHECKPOINT ARGS
  for i in df.iloc:
      if (i[9] < i[10] or i[11] < i[12]):
          print("ERROR: coordinates are wrong!")
      else:
          pass

  df = df.sample(frac=1).reset_index()
  
  train_lim = int(round(len(df) * (1 - test), 0))

  train = df.iloc[0:train_lim]
  test = df.iloc[train_lim + 1:-1]

  train = train.iloc[:,1:]
  test = test.iloc[:,1:]

  return train, test