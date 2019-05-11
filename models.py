import numpy as np
import random
import matplotlib.pyplot as plt
import math
from sklearn import linear_model
import copy
import operator

class LinearRegression():
    def __init__(self):
        self.m = 0
        self.col = 0
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
        self.w = reg.coef_.reshape(self.col, 1)
        total = 0
        for i in range(len(y)):
            total += y[i] - np.dot(x[i].reshape(1, self.col), self.w)
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
        return np.dot(x.reshape(1, self.col), self.w) + self.b

    def predict_all(self, x):
        y = []
        for i in range(len(x)):
            y_predict = self.predict(np.asarray(x[i])).tolist()[0][0]
            y_predict = int(y_predict)
            y.append(y_predict)
        return y

    def error(self, x, y):
        total = 0
        for i in range(len(y)):
            error = pow((y[i] - self.predict(x[i])), 2)
            total += error
        return total / len(y)

    def lassoRidge(self, x, y, lamda):
        x_, y_ = self.transform_data(x, y)
        reg = linear_model.Lasso(alpha = 0.001)
        reg.fit(x_, y_)
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
        new_x = x_.tolist()
        for i in range(del_len):
            for item in new_x:
                del item[index[i]]
            for j in range(del_len):
                index[j] -= 1
        new_x = np.asarray(new_x)
        final_x = np.zeros((self.m, self.col + 1 - del_len))
        for i in range(self.m):
            for j in range(self.col + 1 - del_len):
                if j == 0:
                    final_x[i][0] = 1
                else:
                    final_x[i][j] = new_x[i][j - 1]
        self.ridge(final_x, y_, lamda)
        try:
            weights = self.w.tolist()
        except AttributeError:
            weights = []
            for i in range(self.col - del_len):
                weights.append([0])
        for i in range(del_len):
            weights = weights[:index_2[i]] + [[0]] + weights[index_2[i]:]
        transform_w = []
        for i in range(self.col):
            transform_w.append(weights[i][0])
        self.w = np.asarray(transform_w).reshape(self.col, 1)
        if isinstance(self.b, int):
            return transform_w, self.b
        else:
            return transform_w, self.b.tolist()[0][0]

    def transform_data(self, x, y):
        self.m = len(x)
        self.col = len(x[0])
        # new_x = np.zeros((self.m, self.col))
        # new_y = np.zeros(self.m)
        # for i in range(self.m):
        #     for j in range(self.col):
        #         new_x[i][j] = x[i][j]
        #     new_y[i] = y[i]
        new_x = np.asarray(x)
        new_y = np.asarray(y)
        return new_x, new_y

class knn:
    def classify(self, intX, dataSet, labels, k):
        dataSetSize = dataSet.shape[0]
        diffMat = np.tile(intX, (dataSetSize, 1)) - dataSet
        sqdifMax = diffMat**2
        #caculate distance
        seqDistances = sqdifMax.sum(axis=1)
        distances = seqDistances**0.5
        # print ("distances:",distances)
        sortDistance = distances.argsort()
        # print ("sortDistance:",sortDistance)
        classCount = {}
        for i in range(k):
            voteLabel = labels[sortDistance[i]]
            # print ("the %d voteLabel = %s",i,voteLabel)
            classCount[voteLabel] = classCount.get(voteLabel,0)+1

        sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
        # print ("sortedClassCount:",sortedClassCount)
        return sortedClassCount[0][0]

    def predict(self, intx, dataSet, labels, k):
        y = []
        for i in range(len(dataSet)):
            y_predict = self.classify(intX, dataSet[i], labels, k)
            y_predict = int(y_predict)
            y.append(y_predict)
        return y

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
