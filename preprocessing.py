# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 06:14:55 2019

@author: Medhat
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler

null = 0
class scaler:
  def __init__(self):
      pass
  def standerd_scaler(self, data):
    scaler = StandardScaler()
    scaler.fit(data)
    mean = scaler.mean_
    transform = scaler.transform(data)
    return transform, mean
  def minmax_scaler(self, data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    data_max = scaler.data_max_
    transform = scaler.transform(data)
    return transform, data_max
  def max_scaler(self, data):
    scaler = MaxAbsScaler()
    scaler.fit(data)
    transform = scaler.transform(data)
    return transform  

class encoder:
  def __init__(self):
      pass
  
  def label(self, X_train, X_valid):
      # Get list of categorical variables
      s = (X_train.dtypes == 'object')
      object_cols = list(s[s].index)
      # Make copy to avoid changing original data 
      label_X_train = X_train.copy()
      label_X_valid = X_valid.copy()
      # Apply label encoder to each column with categorical data
      label_encoder = LabelEncoder()
      for col in object_cols:
          label_X_train[col] = label_encoder.fit_transform(X_train[col])
          label_X_valid[col] = label_encoder.transform(X_valid[col])
          
  def ordinal(self, X_train, X_valid):
      # Get list of categorical variables
      s = (X_train.dtypes == 'object')
      object_cols = list(s[s].index)
      #print("Categorical variables:")
      #print(object_cols)
      # Apply ordinal encoder to each column with categorical data
      OH_encoder = OrdinalEncoder( categories='auto')
      OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
      OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
      # ordinal encoding removed index; put it back
      OH_cols_train.index = X_train.index
      OH_cols_valid.index = X_valid.index
      # Remove categorical columns (will replace with ordinal encoding)
      num_X_train = X_train.drop(object_cols, axis=1)
      num_X_valid = X_valid.drop(object_cols, axis=1)
      # Add ordinal encoded columns to numerical features
      OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
      OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
      return OH_X_train,OH_X_valid
  
  def onehat(self, X_train, X_valid):
      # Get list of categorical variables
      s = (X_train.dtypes == 'object')
      object_cols = list(s[s].index)
      #print("Categorical variables:")
      #print(object_cols)
      # Apply one-hot encoder to each column with categorical data
      OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
      OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
      OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
      # One-hot encoding removed index; put it back
      OH_cols_train.index = X_train.index
      OH_cols_valid.index = X_valid.index
      # Remove categorical columns (will replace with one-hot encoding)
      num_X_train = X_train.drop(object_cols, axis=1)
      num_X_valid = X_valid.drop(object_cols, axis=1)
      # Add one-hot encoded columns to numerical features
      OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
      OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
      return OH_X_train,OH_X_valid


class missing_data:
  def __init__(self, missing_data_type):
    self.missing_data_type = missing_data_type
  def run(self, dataframe):
      if self.missing_data_type == "drop":
          dataframe = dataframe.dropna(axis=0)
      elif self.missing_data_type == "mean":
          dataframe = dataframe.fillna(dataframe.mean())
      elif self.missing_data_type == "mode":
          dataframe = dataframe.fillna(dataframe.mode().iloc[0])
      return dataframe