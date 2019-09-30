import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.layers.model import Model
from src.data.dataloader import mini_batch

num_epoch = 100
df = pd.read_csv('data/processed/dataset.csv')
df = df.drop(df.columns[[0]], axis=1)
X = df.drop(['play'], axis=1).values
y = df['play'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.5)
model = Model([6, 6, 3, 2, 1], hidden_layer=2, momentum=0.9, learning_rate=0.01)
for _ in range(num_epoch):
    for batch in mini_batch(X_train, y_train, 4, shuffle=True):
            model(batch[0])
            model.back_prop(batch[1].reshape(-1,1))
pred = model(X_test).flatten()
pred = [1 if x>=0.5 else 0 for x in pred]
print(accuracy_score(pred, y_test))