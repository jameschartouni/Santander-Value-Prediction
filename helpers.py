import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np

#takes a list of column name strings, returns a df with changed columns
#columns are categorical and changed to categorical labels
def label_encoder(df,column_names):
	for column_name in column_names:
		encoder = LabelEncoder()
		a = df[column_name]
		a_encoded = encoder.fit_transform(a)
		df[column_name+"_encoded"] = a_encoded
		print("labels: ")
		print(encoder.classes_)
	return df


#takes a list of column name strings, returns a df with changed columns
#columns are categorical and changed to categorical labels
def label_binarizer(df,column_names):
	for column_name in column_names:
		encoder = LabelBinarizer()
		a = df[column_name]
		a_encoded = encoder.fit_transform(a)
		a_encoded = a_encoded.transpose()
		i = 0
		for col in a_encoded:
			df[column_name+"_encoded" +  str(i)] = col
			i += 1
		#print(list(df))
	return df



#ex: '60 months' --> 60
#converts columns by stripping particular strings and converting resulting string number to float or int
#takes a dictionary of features to strip. Key: the feature, value: the string or character that needs to be deleted
def to_strip(df,strip_dict):
	for key,value in strip_dict.items():
		df[key] = df[key].str.strip(value)
		df[key] = pd.to_numeric(df[key], errors='coerce')
	return df

#takes a dictionary of features with NaNs. Key: feature with NaN values, value: values to replace NaNs
#Value can take three different values: 0, 'med' or 350. 'med' means we fill the NaN with median value of the feature
#350 was used because it was a high number. Ex: "mths_since_last_record" denotes the months since an arrest of some sort. NaNs denoted a clean bill
#350 just is used to show that this individual was never booked
def to_fill_na(df,nan_dict):
	for key,value in nan_dict.items():
		if value == 'med':
			df[key].fillna(value=df[key].median(),inplace=True)
		else:
			df[key].fillna(value=value,inplace=True)
	return df


def rmsle(y_true, y_pred):
    log_error = np.log1p(y_true) - np.log1p(y_pred)
    return np.sqrt( np.square(log_error).mean() )

def rmse(y_true, y_pred):
    error = y_true - y_pred
    return np.sqrt( np.square(log_error).mean() )


import math 

def rmse_2(estimator, X, y0):
    #print(estimator.best_params_)
    y = estimator.predict(X)
    if len(y[y<=-1]) != 0:
        y[y<=-1] = 0.0
    #print("here",y[y<=-1],len(y[y<=-1]))
    assert len(y) == len(y0)
    r = np.sqrt(np.mean(np.power(y-y0, 2)))
    if math.isnan(r):
        print("this is a nan")
        print(scipy.stats.describe(y))
        plt.hist(y, bins=10, color='blue')
        plt.savefig("nan_y.png")
        
    return -r

def rmsle_2(estimator, X, y0):
    #print(estimator.best_params_)
    y = estimator.predict(X)
    if len(y[y<=-1]) != 0:
        y[y<=-1] = 0.0
    #print("here",y[y<=-1],len(y[y<=-1]))
    assert len(y) == len(y0)
    r = np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
    if math.isnan(r):
        print("this is a nan")
        print(scipy.stats.describe(y))
        plt.hist(y, bins=10, color='blue')
        plt.savefig("nan_y.png")
        
    return r


