import numpy as np
import pandas as pd 
from model import Model
from sklearn.metrics import classification_report

# load dataset
df = pd.read_csv('../../data/processed/dataset.csv')

# drop id
df = df.drop(df.columns[[0]], axis=1)

# split manually
y_train = np.array([[0],[1]])
X_train = np.array([[1,1,1,1,1,1],[2,2,2,2,2,2]])

X_test = df.drop('play', 1)[11:14].values
y_test = df.play[11:14].values

# fit and predict
net = Model([6, 6, 1], cost_function="cross_entropy")
net.fit(1, inputs=X_train, labels=y_train, num_epochs=1000, learning_rate=0.1, momentum=0.9)
print(net.predict(X_test))
