import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

import matplotlib
matplotlib.use('TkAgg',force=True)
import numpy as np


def sample_by_class(df, class_column, sample_fraction):
    sample = df.groupby(class_column).apply(lambda x: x.sample(frac=sample_fraction)).reset_index(drop=True)    
    df.reset_index(drop=True, inplace=True)

    return sample
    
def train ():
    for_testing = sample_by_class(df, 'class', 0.2)
    for_training = df.drop(for_testing.index)

    X = for_training.drop(['class'], axis=1)
    classes = for_training['class']
    resampler = RandomOverSampler()
    X, y = resampler.fit_resample(X, classes)

    y = X['gravidade']
    X = X.drop(['gravidade'], axis=1)

    X_test = for_testing.drop(['class', 'gravidade'], axis=1)
    y_test = for_testing['gravidade']

    dtr = DecisionTreeRegressor()

    dtr.fit(X, y)
    y_predict = dtr.predict(X_test)
    mean = mean_squared_error(y_test, y_predict)

    return mean

df = pd.read_csv('input.csv', encoding='latin-1')

for i in range (0, 100):
    number_trees = 100
    medias = []
    for tree_num in range (0, number_trees):
        mean = float(train())
        medias.append(mean)    
    result = f'average: {np.mean(medias)}, lowest: {np.min(medias)}'
