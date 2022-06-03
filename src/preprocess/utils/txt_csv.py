import pandas as pd

def txt_to_csv(path_in="res/Info.txt", path_out="outputs/images_info_0.csv"):
    ''' 
    This function takes a .txt file and saves it as .csv
    '''
    df_info = pd.read_csv(path_in, sep = " ", header = 'infer')
    df_info = df_info.iloc[:, :-1]
    df_info_ = df_info.drop(df_info.index[df_info['CLASS'] == 'NORM'], inplace = True)
    df_info_ = df_info.drop(df_info.columns[[1,3]], axis = 1)
    df_info_.loc[:, 'RADIUS'] = pd.to_numeric(df_info_.RADIUS, downcast='integer')
    df_info_.loc[:, 'X'] = pd.to_numeric(df_info_.X, downcast='integer')
    df_info_.loc[:, 'Y'] = pd.to_numeric(df_info_.Y, downcast='integer')
    df_info__ = df_info_.dropna()
    df_info__.to_csv(path_out, sep = ",")


