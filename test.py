from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class Perceptron:
    '''
        One vs All
    '''
    def __init__(self, nhid, nclass, lr=0.01):
        self.w = np.matrix(np.zeros((nclass, nhid)))
        self.lr = lr

    def fit(self, data, labels, epsilon=0.01, max_iter=4000):
        labels = labels.transpose()
        for i in range(max_iter):
            y_hat = self.sigmoid(self.w @ data.transpose())
            loss = np.mean(np.square(labels-y_hat))
            g = np.multiply(np.multiply((labels-y_hat), y_hat), (1 - y_hat)) * data
            self.w += self.lr * g

            if i % 100 == 0:
                print('Epoch {} loss: {} '.format(i, loss))

            if loss < epsilon:
                break

    def transform(self, data):
        '''vote'''
        y_hat = self.sigmoid(self.w @ data.transpose())
        return np.argmax(y_hat, 0).A.reshape(-1)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def encode_labels(labels):
    n_class = labels.max() + 1
    identity = np.eye(n_class)
    return identity[labels]

def preprocessing(X):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    svd = TruncatedSVD(n_components=300, n_iter=2000, algorithm='arpack')
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

model = Perceptron(nhid=X.shape[1], nclass=n_class, lr=0.01)
model.fit(X_train, y_train)

prediction = model.transform(X_test)
print('test accuracy: ', (prediction == y_test).sum()/len(y_test))

