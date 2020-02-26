# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 06:16:14 2019

@author: Medhat
"""
from sklearn.cluster import KMeans

class clustering:
  def __init__(self):
      pass
  def kmeans(self,X, predection_element, no_of_clusters, random_statval):
      model =KMeans(n_clusters=no_of_clusters, random_state=random_statval)
      model.fit(X)
      labels = model.labels_
      cluster_centers = model.cluster_centers_
      preds = model.predict(predection_element)
      return preds, labels, cluster_centers