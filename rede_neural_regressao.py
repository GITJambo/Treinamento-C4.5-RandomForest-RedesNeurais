import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
# import jamb_log
import logging

import math
import multiprocessing

logpath = "log.txt"
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)
ch = logging.FileHandler(logpath)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)

def log_to_file(message):
   logger.info(message)
   return [message] 

def sample_by_class(df, class_column, sample_fraction):
    return df.groupby(class_column).apply(lambda x: x.sample(frac=sample_fraction)).reset_index(drop=True)

def train_neural_network(layer, activation, max_iter):
    layers = (layer, 10) if type(layer) is int else layer + (10,)


    for_testing = sample_by_class(df, 'class', 0.05)
    for_training = df.drop(for_testing.index)

    X = for_training.drop(['class'], axis=1)
    classes = for_training['class']

    resampler = RandomOverSampler()
    X, y = resampler.fit_resample(X, classes)
    y = X['gravidade']
    X = X.drop(['gravidade'], axis=1)


    X_test = for_testing.drop(['class', 'gravidade'], axis=1)
    y_test = for_testing['gravidade']

    neural = MLPRegressor(hidden_layer_sizes= layers, activation=activation, max_iter=max_iter)

    neural.fit(X, y)

    y_predict = neural.predict(X_test)
    mean = mean_squared_error(y_test, y_predict)
    # print(mean_squared_error(y_test, y_predict))

    return mean

def thread_nn(layer, activation, number_interation, num__neural_networks):
    soma = 0.0
    lowest = math.inf
    for i in range (0, num__neural_networks):
        mean = train_neural_network(layer, activation, number_interation)
        soma += mean
        lowest = lowest if lowest < mean else mean
    result = f'number_interations:{number_interation}, activation: {activation}, layers: {layer}, average: {soma/num__neural_networks}, lowest: {lowest}'
    log_to_file(result)
    

df = pd.read_csv('input.csv', encoding='latin-1')
num__neural_networks =  2


max_iter = [
    500,
    1000,
    3000,
]
activations = ['tanh', 'relu', 'logistic']
layers = [
    (100,100),
    (50,50),
    (500,500),
    ]

threads = []
for layer in layers:
    for number_interation in max_iter:
        for activation in activations:
            thread =  multiprocessing.Process(target=thread_nn, args=(layer, activation, number_interation, num__neural_networks))
            thread.start()
            threads.append(thread)
for t in threads:
    t.join()
