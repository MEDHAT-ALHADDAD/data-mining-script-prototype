# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 06:15:33 2019

@author: Medhat
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class classifier:
  def __init__(self):
    pass

  def bayesian(self,X_train, y_train, predection_element):
      model = GaussianNB()
      model.fit(X_train, y_train)
      preds = model.predict(predection_element)
      return preds

  def decision_tree(self,X_train, y_train, predection_element, random_statval):
      model = DecisionTreeClassifier(random_state=random_statval)
      model.fit(X_train, y_train)
      preds = model.predict(predection_element)
      return preds
 
  def knn(self,X_train, y_train, predection_element, neighbors):
      model = KNeighborsClassifier(n_neighbors=neighbors)
      model.fit(X_train, y_train)
      preds = model.predict(predection_element)
      return preds
  
  def random_forest(self,X_train, y_train, predection_element, estmators, random_stat):
      model = RandomForestClassifier(n_estimators=estmators, random_state=random_stat)
      model.fit(X_train, y_train)
      preds = model.predict(predection_element)
      return preds
