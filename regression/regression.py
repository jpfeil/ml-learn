#!/usr/bin/env python2.7

import math
import quandl
import numpy as np
import pandas as pd
from sklearn import cross_validation, preprocessing, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.00

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x = x[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])


x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)

clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print accuracy

forecast_set = clf.predict(x_lately)

print forecast_set, accuracy, forecast_out
