import numpy as np
import random
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
import copy

class LinearRegression():
    def __init__(self):
        self.w = []
        self.b = 0

    def preDataForLeastSquared(self, x):
        m = len(x)
        new_x = np.zeros((m, 21))
        for i in range(m):
            for j in range(21):
                if j == 0:
                    new_x[i][0] = 1
                else:
                    new_x[i][j] = x[i][j - 1]
        return new_x

    def leastSquared(self, x, y):
        xMat = np.mat(x)
        yMat = np.mat(y).T
        xTx = xMat.T*xMat
        #print(xTx.shape)
        if np.linalg.det(xTx) == 0.0:
            return
        ws = xTx.I*(xMat.T*yMat)
        self.w = ws[1:]
        self.b = ws[0]
        return ws.T

    def ridge(self, x, y, lam):
        xMat = np.mat(x)
        yMat = np.mat(y).T
        xTx = xMat.T * xMat
        if np.linalg.det(xTx) == 0.0:
            return
        ws = (xTx + lam * np.identity(len(x[0]))).I * (xMat.T * yMat)
        self.w = ws[1:]
        self.b = ws[0]
        return ws.T

    def lasso(self, x, y, lamda):
        reg = linear_model.Lasso(alpha = lamda)
        reg.fit(x, y)
        #print(reg.n_iter_)
        #print(reg.coef_)
        vol = len(x[0])
        self.w = reg.coef_.reshape(vol, 1)
        total = 0
        for i in range(len(y)):
            total += y[i] - np.dot(x[i].reshape(1, vol), self.w)
        self.b = total / len(y)
        print('bias is :', self.b)
        print('weights are :', self.w.T)
        return reg.coef_

    def prelasso(self, x):
        vol = len(x[0])
        return np.dot(x.reshape(1, vol), self.w) + self.b

    def errorlasso(self, x, y):
        total = 0
        for i in range(len(y)):
            error = pow((y[i] - self.prelasso(x[i])), 2)
            total += error
        return total / len(y)

    def predict(self, x):
        vol = len(x[0])
        return np.dot(x.reshape(1, vol), self.w) + self.b

    def error(self, x, y):
        total = 0
        for i in range(len(y)):
            error = pow((y[i] - self.predict(x[i])), 2)
            total += error
        return total / len(y)

    def lassoRidge(self, x, y, lamda):
        reg = linear_model.Lasso(alpha = 0.00001)
        reg.fit(x, y)
        #print(reg.n_iter_)
        weights = abs(reg.coef_)
        weights /= sum(weights)
        index = []
        for i in range(len(weights)):
            if weights[i] < 0.01:
                index.append(i)
        index_2 = copy.deepcopy(index)
        #print(index)
        del_len = len(index)
        new_x = x.tolist()
        for i in range(del_len):
            for item in new_x:
                del item[index[i]]
            for j in range(del_len):
                index[j] -= 1
        new_x = np.asarray(new_x)
        vol = len(x[0])
        final_x = np.zeros((1000, vol - del_len))
        for i in range(1000):
            for j in range(vol - del_len):
                if j == 0:
                    final_x[i][0] = 1
                else:
                    final_x[i][j] = new_x[i][j - 1]
        self.ridge(final_x, y, lamda)
        weights = self.w.tolist()
        for i in range(del_len):
            weights = weights[:index_2[i]] + [[0]] + weights[index_2[i]:]
        transform_w = []
        for i in range(vol - 1):
            transform_w.append(weights[i][0])
        self.w = np.asarray(transform_w).reshape(vol - 1, 1)
        return transform_w, self.b.tolist()[0][0]


# if __name__ == '__main__':
#     lams = [0.2]
#     lams = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     lams = [0.00001]
#     lams = [0.00001, 0.0001, 0.001, 0.01, 0.1]
#     errors = []
#     m = 1000
#     for lamda in lams:
#         x, y = generateData(m)
#         #x_ = preDataForLeastSquared(m, x)
#         #print(x)
#         clf = LinearRegression()
#         w, b = clf.lassoRidge(x, y, lamda)
#         print('w(lamda is ' + str(lamda) + '):', w)
#         print('b(lamda is ' + str(lamda) + '):', b)
#         x, y = generateData(m)
#         error = clf.error(x, y)
#         print(error)
#         errors.append(error)
#         errors.append(error.tolist()[0][0])
#         print('True Error(m is ' + str(m) + ', lamda is ' + str(lamda) + '): ', error)
