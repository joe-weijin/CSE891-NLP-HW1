from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

class Perceptron:
    '''
        One vs All
    '''
    def __init__(self, nhid, nclass, lr=0.01, activation='ltu'):
        self.w = np.matrix(np.zeros((nclass, nhid)))
        self.lr = lr
        self.activation = activation

    def fit(self, data, labels, epsilon=0.9):
        false_count = len(labels)
        while(false_count/len(labels) > epsilon):
            false_count = 0
            for x, y in tqdm(zip(data, labels)):
                x = np.matrix(x)
                if self.activation == 'sigmoid':
                    y_hat = self.sigmoid(self.w @ x.transpose())
                else:
                    y_hat = self.sign(self.w @ x.transpose())
                    delta_y = (y.reshape(-1, 1) != y_hat)
                    if delta_y.any():
                        false_count += 1
                        self.w += self.lr * delta_y * y.reshape(-1, 1) * x

    def transform(self, data):
        '''vote'''
        y_hat = self.sign(self.w @ data.transpose())
        return y_hat.max(1)

    def sign(self, x):
        return np.where(x>0, 1, -1)

    def sigmoid(self, x)
        return 1 / (1 + np.exp(-x))

def encode_labels(labels):
    ''' encode label as -1/1, 2 -> [-1, -1, 1, -1, -1, -1...] '''
    n_class = labels.max() + 1
    identity = np.eye(n_class)
    identity = np.where(identity>0, identity, -1)
    return identity[labels]

def preprocessing(X):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    svd = TruncatedSVD(n_components=64, n_iter=2, algorithm='arpack')
    X = svd.fit_transform(X)
    return X

newsgroups_train = fetch_20newsgroups(subset='train')
X_train, y_train = newsgroups_train['data'], newsgroups_train['target']
newsgroups_test = fetch_20newsgroups(subset='test')
X_test, y_test = newsgroups_test['data'], newsgroups_test['target']

n_train = len(X_train)
n_class = max(y_test) + 1

y_train = encode_labels(y_train)
# y_test = encode_labels(y_test)

X = preprocessing(X_train + X_test)
X_train = X[: n_train]
X_test = X[n_train:]

model = Perceptron(nhid=X.shape[1], nclass=n_class)
model.fit(X_train, y_train)

import ipdb
ipdb.set_trace()

prediction = model.transform(X_test)

print('test accuracy: ', (prediction == y_test).sum())

pass

