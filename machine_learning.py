import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import Callback

# customized loss and metric function
from customized_loss_metric import auc_roc, fmeasure
from customized_loss_metric import pair_loss, binary_crossentropy_with_ranking

# for drawing
from IPython.display import clear_output


############################
## data processing functions
############################

def simple_process_prior_fit(df, y_col = 0, y_encoder = None, x_scaler = None, test_size = 0, random_state = 0, 
                            returnEncoder = False, returnScaler = False, returnIndex = False):
    """
    This function process the dataframe prior fitting a model. This simple processing only handle
    encode y labels and scale x. 
    It can return tuple with these elements (in this order): 
    
        x_train, y_train, x_test, y_test, x_train_idx, x_test_idx, y_encoder, x_scaler
        y_encoder and x_scaler are only returned when they're not None
        
    If encoder and scaler are not specified, the function will generate them and fit to the training set
    If you're just process predction set without y label, specify y = -1
    """
    
    if y_col < 0 and test_size > 0:
        raise ValueError('You cannot split data without feature column')
    
    x_train, y_train, x_test, y_test, x_train_idx, x_test_idx =\
        None, None, None, None, None, None
    featureCol = list(range(df.shape[1]))
    
    if y_col >= 0:
        featureCol.remove(y_col)
        y = df.iloc[:,y_col]
        
        # encode y
        if y_encoder is None:
            y_encoder = LabelEncoder()
            y = y_encoder.fit_transform(y)
        else:
            y = y_encoder.transform(y)
        
    x = df.iloc[:,featureCol]

    # split data
    if test_size != 0:
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = test_size, random_state = random_state)
    else:
        x_train = x
        if y_col >= 0:
            y_train = y
    
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

    returnResult = {'x_train' : x_train}
    
    if y_col >= 0:
        returnResult['y_train'] = y_train
    if test_size != 0:
        returnResult['x_test'] = x_test
        returnResult['y_test'] = y_test
    if returnIndex:
        returnResult['x_train_idx'] = x_train_idx
        if test_size != 0:
            returnResult['x_test_idx'] = x_test_idx
    if returnEncoder:
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


##################################
## higher level neural network API
##################################

def simple_ann_binary(input_length, hidden_layers, units, optimizer, loss, 
               dropout = 0, initializer = 'glorot_uniform', b_initializer = 'zeros',  metrics = ['accuracy']):
    classifier = Sequential()
    
    """This function returns a simple ANN, with binary output"""
    
    for i in range(hidden_layers):
        classifier.add(Dense(units = units, kernel_initializer=initializer, 
                             activation= 'relu', input_shape = (input_length,), bias_initializer= b_initializer  ))
        classifier.add(Dropout(dropout))
    
    classifier.add(Dense(units = 1, kernel_initializer=initializer, activation= 'sigmoid', bias_initializer= b_initializer))
    
    classifier.compile(optimizer= optimizer, loss = loss, metrics = metrics)
    return classifier


########################
# Some auxiliary classes
########################

class LossHistory(Callback):
    """This is a Callback class that returns the history of losses for training and validate sets"""
    
    def on_train_begin(self, logs={}):
        self.loss, self.acc, self.auc, self.val_loss, self.val_acc, self.val_auc = [],[],[],[],[],[]
        
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.auc.append(logs.get('auc_roc'))

        if 'val_loss' in logs:
            self.val_loss.append(logs.get('val_loss'))
            self.val_acc.append(logs.get('val_acc'))
            self.val_auc.append(logs.get('val_auc_roc'))
            
    def to_df(self):
        return pd.DataFrame({'loss' : self.loss,
                        'acc' : self.acc,
                        'auc' : self.auc,
                        'val_loss' : self.val_loss,
                        'val_acc' : self.val_acc,
                        'val_auc' : self.val_auc})
    
    def plot(self, columns):
        if columns == 'loss':
            columns = ['loss','val_loss']
        elif columns == 'acc':
            columns = ['acc','val_acc']
        elif columns == 'auc':
            columns = ['auc','val_auc']
        else:
            raise ValueError('columns can only be: loss, acc, or auc')

        history_df = self.to_df()
        g = history_df[columns].plot()
        plt.show()


class PlotLosses(Callback):
    """This is a Callback class that plot losses for training and validation sets in real time
    when fitting the model"""
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()


def test_prediction_withRoc(classifier, x_test, y_test):
    # prediction
    y_pred = classifier.predict(x_test)
    y_pred_TF = (y_pred > 0.5)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_TF)
    cm = pd.DataFrame(cm, columns=['Predicted_false', 'Predicted_true'], index=['Labeled_false','Labeled_true'])
    
    # auc
    y_pred_result = pd.DataFrame({'Label':y_test, 
                                  'predicted_score':y_pred.squeeze(), 
                                  'predicted_true':y_pred_TF.squeeze()})
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_pred_result['Label'], y_pred_result['predicted_score'])
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    acc = (cm.iloc[0,0] + cm.iloc[1,1])/cm.values.sum()
    print(cm)
    print("ROC_AUC: %.3f" % roc_auc)
    print("Accuracy: %.3f" % acc)
    
    # plot ROC
    if roc_auc == roc_auc:
        plt.title('Receiver Operating Characteristic')
        plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.3f'% roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([-0.1,1.05])
        plt.ylim([-0.1,1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    # distribution of prediction score for Benign and Pathogenic variants
    y_pred_result_false = y_pred_result[y_pred_result['Label'] == 0]['predicted_score']
    y_pred_result_true = y_pred_result[y_pred_result['Label'] == 1]['predicted_score']

    if len(y_pred_result_false) > 0:
        y_pred_result_false.plot.kde(color = 'green', label='accu. = %0.3f'% acc)
    if len(y_pred_result_true) > 0:
        y_pred_result_true.plot.kde(color = 'blue', label = '')
    plt.legend(loc='upper center')
    plt.show()



########################################
## filling missing values for dataframes
########################################

def fill_missing_category_by_new_catetory(df, column, trainData = None, fill = 'MISSING'):
    """Fill NaN values for category columns with new category."""
    df.fillna(value = {column:fill},inplace = True)
    
def fill_missing_category_by_random_catetory(df, column, trainData = None):
    """Fill NaN values for category columns with random category from the column."""
    if trainData is None:
        trainData = df
    df[column] = np.where(df[column].isnull(), trainData[column].dropna().sample(len(trainData[column]), replace=True), df[column])     
    
def fill_missing_category_by_most_freq_catetory(df, column, trainData = None):
    """Fill NaN values for category columns with the most frequent category from the column."""
    if trainData is None:
        trainData = df
    df[column] = np.where(df[column].isnull(), trainData[column].dropna().value_counts().idxmax(), df[column])
    
def fill_missing_continous_by_mean(df, column, trainData = None):
    """Fill NaN values for continous columns with mean value."""
    if trainData is None:
        trainData = df
    df[column] = np.where(df[column].isnull(), trainData[column].dropna().mean(), df[column])

def fill_missing_continous_by_median(df, column, trainData = None):
    """Fill NaN values for continous columns with median value."""
    if trainData is None:
        trainData = df
    df[column] = np.where(df[column].isnull(), trainData[column].dropna().median(), df[column])
    

def fill_missing(df, methods, columnNames = None, td = None):
    """Fill all missing values in a dataframe. 
    Available methods: 
        for category: new, random, freq;
        for continuous: mean, median, freq
        To automatically fill missing for all columns with the same method, you just need to 
        specify the method
    """
    
    if columnNames is None:
        columnNames = df.columns
        methods = [methods] * len(columnNames)
    
    for column, method in zip(columnNames, methods):
        if column not in df.columns:
            raise ValueError("Invalid column name %s" % column)
        
        if method == 'new':
            f = fill_missing_category_by_new_catetory
        elif method == 'random':
            f = fill_missing_category_by_random_catetory
        elif method == 'freq':
            f = fill_missing_category_by_most_freq_catetory
        elif method == 'mean':
            f = fill_missing_continous_by_mean
        elif method == 'median':
            f = fill_missing_continous_by_median
        else:
            raise ValueError("Invalid method %s for column %d" % (method, column))
        f(df, column, trainData = td)


