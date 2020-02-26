# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 06:15:54 2019

@author: Medhat
"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class regression:
  def __init__(self):
    pass

  def linear_reg(self,X_train, y_train, predection_element):
      model = LinearRegression()
      model.fit(X_train, y_train)
      score = model.score(X_train, y_train)
      coef = model.coef_
      intercept = model.intercept_
      preds = model.predict(predection_element)
      return preds, coef, score, intercept
  
  def polynomial_reg(self,X_train, y_train, predection_element, degree_of_poly):
      poly = PolynomialFeatures(degree=degree_of_poly)
      X_ = poly.fit_transform(X_train)
      predict_ = poly.fit_transform(predection_element)
      model = LinearRegression()
      model.fit(X_, y_train)
      preds = model.predict(predict_)
      return preds
  
  def decision_tree(self,X_train, y_train, predection_element, random_statval):
      model = DecisionTreeRegressor(random_state=random_statval)
      model.fit(X_train, y_train)
      preds = model.predict(predection_element)
      return preds
      
  def knn(self,X_train, y_train, predection_element, neighbors):
      model = KNeighborsRegressor(n_neighbors=neighbors)
      model.fit(X_train, y_train)
      preds = model.predict(predection_element)
      return preds

  def random_forest(self,X_train, y_train, predection_element, estmators, random_stat):
      model = RandomForestRegressor(n_estimators=estmators, random_state=random_stat)
      model.fit(X_train, y_train)
      preds = model.predict(predection_element)
      return preds  
    