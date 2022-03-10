"""
@Time ： 2020/8/15 10:19
@Author ：KI 
@File ：bp_mnist.py
@Motto：Hungry And Humble

"""
import numpy as np
import torchvision
import torchvision.transforms as transforms

'''
input layer：784 nodes; hidden layer：three hidden layers with 20 nodes in each layer
output layer：10 nodes

'''


class BP:
    def __init__(self):
        self.input = np.zeros((100, 784))  # 100 samples per round
        self.hidden_layer_1 = np.zeros((100, 20))
        self.hidden_layer_2 = np.zeros((100, 20))
        self.hidden_layer_3 = np.zeros((100, 20))
        self.output_layer = np.zeros((100, 10))
        self.w1 = 2 * np.random.random((784, 20)) - 1  # limit to (-1, 1)
        self.w2 = 2 * np.random.random((20, 20)) - 1
        self.w3 = 2 * np.random.random((20, 20)) - 1
        self.w4 = 2 * np.random.random((20, 10)) - 1
        self.error = np.zeros(10)
        self.learning_rate = 0.15

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deri(self, x):
        return x * (1 - x)

    def forward_prop(self, data, label):  # label:100 X 10,data: 100 X 784
        self.input = data
        self.hidden_layer_1 = self.sigmoid(np.dot(self.input, self.w1))
        self.hidden_layer_2 = self.sigmoid(np.dot(self.hidden_layer_1, self.w2))
        self.hidden_layer_3 = self.sigmoid(np.dot(self.hidden_layer_2, self.w3))
        self.output_layer = self.sigmoid(np.dot(self.hidden_layer_3, self.w4))
        # error
        self.error = label - self.output_layer
        return self.output_layer

    def backward_prop(self):
        output_diff = self.error * self.sigmoid_deri(self.output_layer)
        hidden_diff_3 = np.dot(output_diff, self.w4.T) * self.sigmoid_deri(self.hidden_layer_3)
        hidden_diff_2 = np.dot(hidden_diff_3, self.w3.T) * self.sigmoid_deri(self.hidden_layer_2)
        hidden_diff_1 = np.dot(hidden_diff_2, self.w2.T) * self.sigmoid_deri(self.hidden_layer_1)
        # update
        self.w4 += self.learning_rate * np.dot(self.hidden_layer_3.T, output_diff)
        self.w3 += self.learning_rate * np.dot(self.hidden_layer_2.T, hidden_diff_3)
        self.w2 += self.learning_rate * np.dot(self.hidden_layer_1.T, hidden_diff_2)
        self.w1 += self.learning_rate * np.dot(self.input.T, hidden_diff_1)


# from torchvision load data
def load_data():
    datasets_train = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor())
    datasets_test = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    data_train = datasets_train.data
    X_train = data_train.numpy()
    X_test = datasets_test.data.numpy()
    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))
    Y_train = datasets_train.targets.numpy()
    Y_test = datasets_test.targets.numpy()

    real_train_y = np.zeros((60000, 10))
    real_test_y = np.zeros((10000, 10))
    # each y has ten dimensions
    for i in range(60000):
        real_train_y[i, Y_train[i]] = 1
    for i in range(10000):
        real_test_y[i, Y_test[i]] = 1
    index = np.arange(60000)
    np.random.shuffle(index)
    # shuffle train_data
    X_train = X_train[index]
    real_train_y = real_train_y[index]

    X_train = np.int64(X_train > 0)
    X_test = np.int64(X_test > 0)

    return X_train, real_train_y, X_test, real_test_y


# train
def bp_network():
    nn = BP()
    X_train, Y_train, X_test, Y_test = load_data()
    batch_size = 100
    epochs = 6000
    for epoch in range(epochs):
        start = (epoch % 600) * batch_size
        end = start + batch_size
        # print(start, end)
        nn.forward_prop(X_train[start: end], Y_train[start: end])
        nn.backward_prop()

    return nn


def bp_test():
    nn = bp_network()  # model
    sum = 0
    X_train, Y_train, X_test, Y_test = load_data()
    # test：
    for i in range(len(X_test)):
        res = nn.forward_prop(X_test[i], Y_test[i])
        res = res.tolist()
        index = res.index(max(res))  # maximal probability
        if Y_test[i, index] == 1:
            sum += 1

    print('accuracy：', sum / len(Y_test))


if __name__ == '__main__':
    bp_test()
