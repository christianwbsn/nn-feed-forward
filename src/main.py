import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.layers.model import Model
from src.data.dataloader import mini_batch


def train(inputs,labels):
    parser = argparse.ArgumentParser(description='Mini-batch Gradient Descent')
    parser.add_argument('--epoch', default=100, type=int,
                    help='number of iterations')
    parser.add_argument('--b_size', default=5, type=int,
                    help='batch_size')
    parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate')
    parser.add_argument('--m', default=0.9, type=float,
                    help='momentum')
    args = parser.parse_args()
    model = Model([6, 6, 3, 1], hidden_layer=2, momentum=args.m, learning_rate=args.lr)
    for _ in range(args.epoch):
        for batch in mini_batch(inputs, labels, args.b_size, shuffle=True):
            model(batch[0])
            model.back_prop(batch[1].reshape(-1,1))
    return model

def predict(model, inputs, threshold=0.5):
    pred = model(inputs).flatten()
    pred = [1 if x>=threshold else 0 for x in pred]
    return pred


if __name__ == '__main__':
    df = pd.read_csv('data/processed/dataset.csv')
    df = df.drop(df.columns[[0]], axis=1)
    X = df.drop(['play'], axis=1).values
    y = df['play'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
    model = train(X_train, y_train)
    prediction = predict(model, X_test)
    print(classification_report(prediction, y_test))