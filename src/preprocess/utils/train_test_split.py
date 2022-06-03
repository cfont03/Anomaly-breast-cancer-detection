def train_test_split(df, test = 0.3, path_train = 'outputs/train.csv', path_test = 'outputs/test.csv'):

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
    8, 9, 10, 11: coord of boundary box in xmin, xmax, ymin, ymax order
  
  test: % of test data of all dataframe


  '''
  
  ### CHECKPOINT ARGS
  for i in df.iloc:
      if (i[8] > i[9] or i[10] > i[11]):
          print("ERROR: coordinates are wrong!")
      else:
          pass

  df = df.sample(frac=1).reset_index()
  
  train_lim = int(round(len(df) * (1 - test), 0))

  train = df.iloc[0:train_lim]
  test = df.iloc[train_lim + 1:-1]

  train = train.iloc[:,1:]
  test = test.iloc[:,1:]

  train.to_csv(path_train, sep = ",")
  test.to_csv(path_test, sep = ",")

  return train, test