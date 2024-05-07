#--just practice--
from src.function import *


class TwoLayerNetwork:
    def __init__(self, input_num, hidden_num, output_num, initWeight=0.01):

        self.params = {}
        self.params['W1'] = np.random.rand(input_num, hidden_num)
        self.params['b1'] = np.random.rand(hidden_num)
        self.params['W2'] = np.random.rand(hidden_num, output_num)
        self.params['b2'] = np.random.rand(output_num)

    def predict(self, x):
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = sigmoid(a2)
        return y

# according to different dataModel, choosing proper cost function
    def loss_function(self, x, t):
        y = self.preditc(x)
        return mean_cost_function(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        if np.size(y, 1) > 1:
            acc = np.sum(np.argmax(y, 1) == np.argmax(t, 1))/np.size(y, 0)
        else:
            acc = np.sum(y == t)/np.size(y, 0)
        return acc

    def sigmoid_bp(self, x):
        return sigmoid(x)*(1 - sigmoid(x))

    def gradient(self, x, t):
        # forward p
        a1 = np.dot(x, self.params['W1']) + self.params['b1']
        z1 = sigmoid(a1)
        a2 = np.dot(z1, self.params['W2']) + self.params['b2']
        y = sigmoid(a2)

        # backward p
        dloss = y - t
        dz2 = dloss * self.sigmoid_bp(a2)
        da2 = np.dot(dz2, self.params['W2'].t)
        dz1 = da2 * self.sigmoid_bp(a1)

        grad = {}
        grad['W1'] = np.dot(x.t, dz1)
        grad['b1'] = np.sum(dz1, 1)
        grad['W2'] = np.dot(z1.t, dz2)
        grad['b2'] = np.sum(dz2, 1)
        return grad





