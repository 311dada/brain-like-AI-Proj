'''
@Author: Da Ma
@Date: 2019-12-18 15:10:02
@LastEditTime : 2019-12-26 10:01:13
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Proj/src/main.py
'''
import mxnet as mx
from mxnet import autograd, init
from mxnet.gluon import nn
import mxnet.ndarray as nd
from utils.data_gen import generate_data
from sklearn.metrics import classification_report

'''
class squareNN(nn.Block):
    def __init__(self, inunits, units, prefix=None, params=None):
        """init function.
        y = U * X ^ 2 + V * X + b

        Args:
            inunits (int): input units number
            units (int): output units number
            prefix ([type], optional): [description]. Defaults to None.
            params ([type], optional): [description]. Defaults to None.
        """
        super(squareNN, self).__init__(prefix=prefix, params=params)
        self.U = self.params.get('U', shape=(inunits, units))
        self.V = self.params.get('V', shape=(inunits, units))
        self.bias = self.params.get('bias', shape=(1, units))

    def forward(self, X):
        linear = nd.dot(nd.square(X), self.U.data()) + \
            nd.dot(X, self.V.data()) + self.bias.data()
        return nd.relu(linear)
'''


class NN(nn.Block):
    def __init__(self, layer, hidden=100, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.blk = nn.Sequential()

        # self.blk.add(squareNN(2, hidden))
        for i in range(layer - 1):
            self.blk.add(nn.Dense(hidden, activation='relu', flatten=False))
            # self.blk.add(squareNN(hidden, hidden))

        self.output = nn.Dense(2, flatten=False)
        # self.output = squareNN(hidden, 2)

    def forward(self, X, activation=None):
        if activation is not None:
            return activation(self.output(self.blk(X)))
        else:
            return self.output(self.blk(X))


class FNN():
    def __init__(self,
                 train_num,
                 rate,
                 layer,
                 hidden,
                 batch_size,
                 train_circle_num,
                 train_r_rate,
                 train_offset,
                 train_start,
                 test_circle_num,
                 test_r_rate,
                 test_offset,
                 test_start,
                 learning_rate=0.1,
                 iteration_num=10,
                 params=None):
        """the init function for FNN

        Args:
            train_num (int): the sample number of the train data
            rate (double): the rate of test data for the train data
            layer (int): the depth of the FNN
            hidden (int): the hidden size of the FNN
            batch_size (int): batch size
            circle_num (int): the circle number of the spiral
            r_rate (double): to control the radius
            offset (int): to control the radius
            start (double): the start position
            iteration_num (int, optional): the iteration number of the training. Defaults to 10.
            learning_rate (double, optional): the learning rate. Defaults to 0.1.
            params: (str, optional): the params path. Defaults to None.
        """
        self.model = NN(layer, hidden)
        self.batch_size = batch_size
        self.iteration_num = iteration_num
        self.learning_rate = learning_rate

        if params is not None:
            self.model.load_parameters(params)

        # generate data
        train_num = train_num // 2
        test_num = int(train_num * rate)
        self.train_data, self.train_labels = generate_data(
            train_num, train_circle_num, train_r_rate, train_offset,
            train_start)

        self.test_data, self.test_labels = generate_data(
            test_num, test_circle_num, test_r_rate, test_offset, test_start)

        self.train_data = nd.array(self.train_data)
        self.train_labels = nd.array(self.train_labels,
                                     dtype="int32")
        self.test_data = nd.array(self.test_data)
        self.test_labels = nd.array(self.test_labels,
                                    dtype="int32")

    def train(self, loss="MSE", save_path=None):
        """the train function

        Args:
            loss (str, optional): the loss function to use, C1 or MSE. Defaults to "MSE".
            save_path (str, optional): the saving params path. Defaults to None.
        """
        dataset = mx.gluon.data.ArrayDataset(self.train_data,
                                             self.train_labels)
        self.model.collect_params().initialize(init=init.Xavier())
        trainer = mx.gluon.Trainer(self.model.collect_params(), 'sgd',
                                   {'learning_rate': self.learning_rate})

        # train
        loss_history = []
        for epoch in range(self.iteration_num):
            train_loss = 0.0
            dataloader = mx.gluon.data.DataLoader(dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True)

            for train_data, train_labels in dataloader:
                with autograd.record():
                    output = self.model(train_data)
                    if loss == "MSE":
                        output = nd.sigmoid(output)
                        Loss = mx.gluon.loss.L2Loss()
                        Loss = Loss(output,
                                    nd.cast(train_labels, dtype="float32"))
                    else:
                        Loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(
                            sparse_label=False)
                        Loss = Loss(output,
                                    nd.cast(train_labels, dtype="float32"))

                Loss.backward()
                loss_history.append(Loss.mean().asscalar())
                trainer.step(self.batch_size)

                train_loss += Loss.mean().asscalar()
            print(f"epoch {epoch}:\n")
            print(f"trainning loss: {train_loss / len(dataloader)}\n")

            if epoch % 10 == 0:
                print("trainning report:")
                self.evaluate(self.model(self.train_data), self.train_labels)
                # print()
                # print("testing report:")
                # self.evaluate(self.model(self.test_data), self.test_labels)
            print("=" * 100)

        if save_path is not None:
            self.model.save_parameters(save_path)

        return loss_history

    def predict(self, X):
        """the predict function

        Args:
            X (ndarray or nparray): the data to predict
        """
        X = nd.array(X)
        Y = nd.argmax(self.model(X), axis=1)
        return nd.cast(Y, "int32")

    def evaluate(self, pred, label):
        """evaluate the prediction of the model

        Args:
            pred (ndarray): the prediction ndarray
            label (ndarray): the label ndarray
        """
        y_pred = nd.cast(nd.argmax(pred, axis=1), dtype="int32").asnumpy()
        y_true = nd.cast(nd.argmax(nd.cast(label, dtype="float32"), axis=1),
                         dtype="int32").asnumpy()
        target_names = ["spiral0", "spiral1"]

        print(classification_report(y_true, y_pred, target_names=target_names))
