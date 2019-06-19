#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:07:49 2019

@author: pete
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os


def cardinality_check(df):
    dtypes = pd.DataFrame(df.dtypes)
    n_unique = pd.DataFrame(df.nunique())
    n_unique['pct_unique'] = n_unique/len(df)
    n_unique.sort_values(by='pct_unique', ascending=False, inplace=True)
    cardinality_df = n_unique.join(dtypes, lsuffix='_l', rsuffix='_r')
    cardinality_df.columns = ['n_unique', 'pct_unique', 'data_type']
    return cardinality_df


#Separate features by data type
def segment_features(df, pct_unique_cutoff):
    c_df = cardinality_check(df)
    object_features = c_df[c_df.data_type == 'object']
    numerical_features = c_df[~np.isin(
            c_df.index.values, object_features.index.values
            )]
    return {
        "cat" : object_features.index.values,
        "num" : numerical_features[
                    numerical_features.pct_unique > pct_unique_cutoff
                ].index.values,
        "ord" : numerical_features[
                    numerical_features.pct_unique <= pct_unique_cutoff
                ].index.values
            }

#Get data
ames = pd.read_csv('Data/train.csv')
ames.drop('Id', axis=1, inplace=True)
y = ames.pop('SalePrice')

#Define feature list for preprocessing
features = segment_features(ames, .015)
all_features = list(itertools.chain.from_iterable(features.values()))


#Create pipelines for tree based models
numeric_transform = Pipeline(steps=[
        ('impute', IterativeImputer(add_indicator=True))
        ])
    
categorical_transform = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent', add_indicator=True)),
        ('one_hot', OneHotEncoder(handle_unknown='ignore'))
        ])
   
preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_transform, features['num']),
        ('categorical', categorical_transform, np.concatenate(
                (features['cat'], features['ord']),
                )
    )])
    
#Separate data for baseline performance calculation
X_60pct = ames.sample(int(.6*ames.shape[0]), random_state=33)
y_60pct = y[X_60pct.index.values]
X_train, X_test, y_train, y_test = train_test_split(X_60pct, y_60pct, test_size=.1)

#Fit model
rf = RandomForestRegressor(n_estimators=100, max_features="sqrt", oob_score=True)
model = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('regression', rf)
        ])
model.fit(X_train, y_train)
model.score(X_test, y_test)

    