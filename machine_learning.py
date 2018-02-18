import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


############################
## data processing functions
############################

def simple_process_prior_fit(df, y_col = 0, y_encoder = None, x_scaler = None, test_size = 0, random_state = 0, 
                            returnEncoder = False, returnScaler = False, returnIndex = False):
    """
    This function process the dataframe prior fitting a model. This simple processing only handle
    encode y labels and scale x. 
    It will returns tuple with file elements: 
    
        x_train, y_train, x_test, y_test (x_train_idx, x_test_idx, y_encoder, x_scaler)
        y_encoder and x_scaler are only returned when they're not None
        
    If encoder and scaler are not specified, the function will generate them and fit to the training set
    """
    
    featureCol = list(range(df.shape[1]))
    featureCol.remove(y_col)
    
    x = df.iloc[:,featureCol]
    y = df.iloc[:,y_col]
    
    # encode y
    if y_encoder is None:
        y_encoder = LabelEncoder()
        y = y_encoder.fit_transform(y)
    else:
        y = y_encoder.transform(y)

    # split data
    if test_size != 0:
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = test_size, random_state = random_state)
    else:
        x_train, y_train = x, y
        x_test, y_test, x_test_idx = None, None, None
    
    # get index before transform the data
    if returnIndex:
        x_train_idx = x_train.index
        if test_size != 0:
            x_test_idx = x_test.index
        
    # feature scaling
    if x_scaler is None:
        x_scaler = StandardScaler()
        x_train = x_scaler.fit_transform(x_train)
    else:
        x_train = x_scaler.transform(x_train)
    
    if test_size != 0:
        x_test = x_scaler.transform(x_test)

    returnResult = {'x_train' : x_train, 
                    'y_train' : y_train,
                    'x_test' : x_test,
                    'y_test' : y_test}
    if returnIndex:
        returnResult['x_train_idx'] = x_train_idx
        returnResult['x_test_idx'] = x_test_idx
    if returnreturnEncoder:
        returnResult['y_encoder'] = y_encoder
    if returnScaler:
        returnResult['x_scaler'] = x_scaler
    
    return returnResult


##################################################################
# up-sampling of minority class or down-sampling of majority class
# there's a package called imbalanced-learn
##################################################################

def resampling(df, class_column, down = True):
    # disadvantage is to introduce overfitting when upsampling
    # and lose data, or introduced bias when downsampling
    
    from sklearn.utils import resample
    class_series = df[class_column]
    uniq_class = set(class_series.unique())
    
    # find the most / least abundant class
    counts = class_series.value_counts()
    if down:
        m_count = class_series.value_counts().min()
        replace = False
    else:
        m_count = class_series.value_counts().max()
        replace = True
        
    m_Class = counts.index[np.where(counts == m_count)[0][0]]
    
    result = df[class_series == m_Class]
    
    for cla in (uniq_class - {m_Class}):
        if counts[cla] != m_count:
            cla_resampled = resample(df[class_series == cla], replace=replace,
                                     n_samples=m_count, random_state=0)
            result = pd.concat([result, cla_resampled])
        else:
            result = pd.concat([result, df[class_series == cla]])
    return result

