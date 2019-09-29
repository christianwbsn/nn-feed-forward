import numpy as np
import pandas as pd 
from model import Model
from sklearn.metrics import classification_report

# load dataset
df = pd.read_csv('../../data/processed/dataset.csv')

# drop id
df = df.drop(df.columns[[0]], axis=1)

# split manually
y_train = df.play[0:11].values
X_train = df.drop('play', 1)[0:11].values

X_test = df.drop('play', 1)[11:14].values
y_test = df.play[11:14].values

# fit and predict
net = Model([6, 4, 1], cost_function="mean_squared")
net.fit(1, inputs=X_train, labels=y_train, num_epochs=10, learning_rate=0.001)

y_pred = net.predict(X_test)

# classification report
print(classification_report(y_test, y_pred))
