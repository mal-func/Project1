# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 17:22:28 2021

@author: Harsh
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import scipy.stats as ss
from sklearn.neighbors import KNeighborsClassifier
import random 


"""
input parameters: n = number of points required
returns: outcomes, points(predictors)

"""
def synthetic_data(n):
    points = np.concatenate((ss.norm(0, 1).rvs((n, 2)), ss.norm(1, 1).rvs((n, 2))))
    outcomes = np.concatenate((np.repeat(0,n), np.repeat(1, n)))
    return (points, outcomes)



"""
input parameters: point1, point2
returns: Euclidean distance between point1 and point2

"""
def dist(x1,x2):    
    return np.sqrt(np.sum(np.power((x1 - x2),2)))



"""
input parameters: points(predictors), point p, k
returns: sorted indices

"""
def nearest_neighbour(points, p, k=5):   
    distances = np.zeros(len(points))
    for i in range(len(distances)):
        distances[i] = dist(p, points[i])     
    ind = np.argsort(distances)
    return ind[:k]



"""
input parameters: outcomes, points(predictors), point p, k
returns: the class a point belongs to

"""
def predict_class(outcomes, points, p, k):
    ind = nearest_neighbour(points, p, k)
    return majority_votes(outcomes[ind])       



"""
input parameters: votes
returns: majority votes

"""
def majority_votes(votes):    
    votes_dict = {}
    winners = []
    for vote in votes:
        if vote in votes_dict:
            votes_dict[vote] += 1
        else:
            votes_dict[vote] = 1
    max_value = max(votes_dict.values())
    for elem, counts in votes_dict.items():
        if counts == max_value:
            winners.append(elem)    
    return random.choice(winners)



"""
creates a prediction grid
input parameters: tuple limits, predictors(points), outcomes, h, k
returns: prediction grid, meshgrid xx and yy

"""
def make_prediction_grid(limits, predictors, outcomes, h, k):
    
    x_min, x_max, y_min, y_max = limits
    xs = np.arange(x_min, x_max, h)
    ys = np.arange(y_min, y_max, h)
    xx, yy = np.meshgrid(xs, ys)
    
    prediction_grid = np.zeros(xx.shape)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            p = np.array([x,y])
            prediction_grid[j,i] = predict_class(outcomes, predictors, p, k) 
    
    return prediction_grid, xx, yy



"""
plots the prediction grid on a graph
input parameters: meshgrid xx & yy, prediction grid, predictors(points), outcomes
returns: null

"""
def plot_prediction_grid (xx, yy, prediction_grid, predictors, outcomes):
    background_colormap = ListedColormap (["hotpink", "lightskyblue", "yellowgreen"])
    observation_colormap = ListedColormap (["red", "blue", "green"])
    plt.figure(figsize =(10,10))
    plt.pcolormesh(xx, yy, prediction_grid, cmap = background_colormap, alpha = 0.5)
    plt.scatter(predictors[:,0], predictors [:,1], c = outcomes, cmap = observation_colormap, s = 50)
    plt.xlabel('Variable 1'); plt.ylabel('Variable 2')
    plt.xticks(()); plt.yticks(())
    plt.xlim (np.min(xx), np.max(xx))
    plt.ylim (np.min(yy), np.max(yy))
    plt.show()

training_predictors, training_outcomes = synthetic_data(250)
testing_predictors, testing_outcomes = synthetic_data(10)
training_predictions = [predict_class(training_outcomes, training_predictors, p, k=5) for p in training_predictors]
testing_predictions = [predict_class(training_outcomes, training_predictors, p, k=5) for p in testing_predictors]

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(training_predictors, training_outcomes)
scikit_training_predictions = knn.predict(training_predictors)
scikit_testing_predictions = knn.predict(testing_predictors)

print(f"training predictions: {np.mean(training_predictions == training_outcomes)}")
print(f"testing predictions: {np.mean(testing_predictions == testing_outcomes)}") 
print(f"scikit training predictions: {np.mean(scikit_training_predictions == training_outcomes)}") 
print(f"scikit testing predictions: {np.mean(scikit_testing_predictions == testing_outcomes)}") 

prediction_grid, xx, yy = make_prediction_grid((-3,4,-3,4), training_predictors, training_outcomes, 0.1, 5)
plot_prediction_grid(xx, yy, prediction_grid, training_predictors, training_outcomes)



       