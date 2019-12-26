'''
@Author: Da Ma
@Date: 2019-12-19 14:45:40
@LastEditTime : 2019-12-26 10:01:02
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Proj/src/main.py
'''

from model import FNN
import os
from utils.Plot import plot_decision_boundary
import numpy as np
import matplotlib.pyplot as plt
import mxnet

if __name__ == "__main__":
    SEED = 5
    train_num = 1000
    rate = 0.25
    layer = 3
    hidden = 100
    batch_size = 32
    train_circle_num = 2
    train_r_rate = 0.8
    train_offset = 100
    train_start = 0.5
    test_circle_num = 7
    test_r_rate = 0.8
    test_offset = 200
    test_start = -9
    learning_rate = 0.1
    epochs = 10 * 2000 // train_num
    save_path = os.path.join(
        "../params",
        f"train_num_{train_num}-layer_{layer}-epochs_{epochs}.params")

    np.random.seed(SEED)
    mxnet.random.seed(SEED)

    fnn = FNN(train_num, rate, layer, hidden, batch_size, train_circle_num,
              train_r_rate, train_offset, train_start, test_circle_num,
              test_r_rate, test_offset, test_start, learning_rate, epochs)

    loss_history = fnn.train(loss="C1", save_path=save_path)

    # print(fnn.train_data.asnumpy())
    # print(np.argmax(fnn.train_labels.asnumpy(), axis=1))
    plot_decision_boundary(fnn, fnn.train_data.asnumpy(),
                           np.argmax(fnn.train_labels.asnumpy(), axis=1))

    cnt, loss_avg = 0, []
    while cnt < len(loss_history):
        loss_avg.append(
            sum(loss_history[cnt: min(cnt + 100, len(loss_history))]) / 100)
        cnt += 100

    plt.plot(loss_avg)
    plt.show()
    # train_data = fnn.train_data.asnumpy()
    # train_labels = np.argmax(fnn.train_labels.asnumpy(), axis=1)

    # spiral0 = train_data[train_labels==0]
    # spiral1 = train_data[train_labels==1]

    # plt.scatter(spiral0[:,0], spiral0[:,1])
    # plt.scatter(spiral1[:,0], spiral1[:,1])
    # plt.show()
