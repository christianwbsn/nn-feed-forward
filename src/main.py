import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.layers.model import Model
from src.layers.loss import MeanSquaredError
from src.data.dataloader import mini_batch
from sklearn.preprocessing import StandardScaler


def train(inputs,labels):
    parser = argparse.ArgumentParser(description='Mini-batch Gradient Descent')
    parser.add_argument('--epoch', default=60, type=int,
                    help='number of iterations')
    parser.add_argument('--b_size', default=3, type=int,
                    help='batch_size')
    parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate')
    parser.add_argument('--m', default=0.9, type=float,
                    help='momentum')
    args = parser.parse_args()
    input_layer = inputs.shape[1]
    model = Model([input_layer,8, 1], hidden_layer=1, momentum=args.m, learning_rate=args.lr)
    criterion = MeanSquaredError()
    for i in range(args.epoch):
        running_loss = 0.0
        print("Beforeee")
        print(model.ff[0].weight)
        print()
        print(model.ff[1].weight)
        for batch in mini_batch(inputs, labels, args.b_size, shuffle=True):
            model(batch[0])
            model.back_prop(batch[1].reshape(-1,1))
            running_loss = criterion(model(batch[0]), batch[1].reshape(-1,1))
        print("After")
        print(model.ff[0].weight)
        print()
        print(model.ff[1].weight)
        loss = running_loss / args.b_size
        print("Epoch {0}/{1} :{2}".format(i+1, args.epoch, loss))
    return model


def predict(model, inputs, threshold=0.5):
    pred = model(inputs).flatten()
    print("Pred:",pred)
    pred = [1 if x>=threshold else 0 for x in pred]
    return pred

if __name__ == '__main__':
    df = pd.read_csv('data/processed/train_titanic.csv')
    X = df.drop(['Survived'], axis=1).values
    y = df['Survived'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3, stratify=y)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model = train(X_train, y_train)
    prediction = predict(model, X_test)
    print(prediction)
    print(classification_report(prediction, y_test))