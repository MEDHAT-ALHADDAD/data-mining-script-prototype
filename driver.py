# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 06:16:45 2019

@author: Medhat
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 02:54:08 2019

@author: Medhat
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import encoder, missing_data, scaler
from classification import classifier
from sklearn.metrics import mean_absolute_error
from sklearn import metrics

# Read the data
data = pd.read_csv('./melb_data.csv')

##### 1- preprocessing missing data
# missing data
md = missing_data("mode")
data = md.run(data)


# Separate target from predictors    !!!!!!! we change both Price to the col you want to predict
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

#print(X_train.head())

##### 1- preprocessing encoding
encoder = encoder()
OH_X_train, OH_X_valid = encoder.ordinal(X_train, X_valid)


#### 1- preprocessing scaling

scaler = scaler()
datat = scaler.standerd_scaler(OH_X_train)
datat = scaler.standerd_scaler(OH_X_valid)

##### 2- modeling
cla = classifier()
preds = cla.random_forest(OH_X_train, y_train, OH_X_valid, 10, 0)
print(preds)
print(mean_absolute_error(y_valid, preds))

#Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_valid, preds) * 100, "%")








