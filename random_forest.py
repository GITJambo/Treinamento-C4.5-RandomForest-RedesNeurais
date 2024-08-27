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

df = pd.read_csv('input.csv', encoding='latin-1')
    
def train (ccp_alpha, for_training):
    X = for_training.drop(['class'], axis=1)
    classes = for_training['class']

    resampler = SMOTE()
    X, y = resampler.fit_resample(X, classes)

    y = X['gravidade']
    X = X.drop(['gravidade'], axis=1)

    dtr = DecisionTreeRegressor(ccp_alpha=ccp_alpha)

    dtr.fit(X, y)
    return dtr


def forest(num_arvores, ccp):
    ccp_fraction = ccp if ccp == 0 else 1 / ccp

    for_validation = sample_by_class(df, 'class', 0.2)
    for_training = df.drop(for_validation.index)

    X_valid = for_validation.drop(['class', 'gravidade'], axis=1)
    y_valid = for_validation['gravidade']

    modelos = []
    for i in range (0, num_arvores):
        modelos.append(train(ccp_fraction, for_training))

    avg_arr = []
    actual = []
    for i in range(int(X_valid.shape[0])):
        index = X_valid.index[i]
        actual.append(y_valid[index])

        resultados = []
        for modelo in modelos:
            prediction = modelo.predict(X_valid.iloc[[i]])
            resultados.append(float(prediction.item()))
        avg_arr.append(np.average(resultados))
        
    ccp_log = str(ccp) if ccp == 0 else f'1 / {ccp}'
    result = f'trees: {num_arvores}, ccp: {ccp_log}, average: {mean_squared_error(avg_arr, actual)}'
    print(result)


num_arvores = [20, 30, 50, 80, 100]
for quantidade in num_arvores:
    forest(quantidade, 0)
