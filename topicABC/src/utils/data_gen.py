'''
@Author: Da Ma
@Date: 2019-12-18 14:30:44
@LastEditTime : 2019-12-26 10:01:25
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Proj/src/utils/data_gen.py
'''

import numpy as np
import math
import matplotlib.pyplot as plt


def generate_data(sample_num, circle_num, r_rate, offset, start):
    """generate the double spiral data according to the formula mentioned in the report.

    Args:
        sample_num (int): the number of samples to generate
        circle_num (int): the circle number of the spiral
        r_rate (double): to control the radius
        offset (int): to control the radius
        start (double): the start position

    Return:
        two spirals numpy array and corresponding label .
    """
    n = np.arange(sample_num)
    alpha = math.pi * n / (sample_num / (2 * circle_num))
    r = r_rate * (sample_num + offset - n) / (sample_num + offset - 1)
    x1 = r * np.sin(alpha) + start
    y1 = r * np.cos(alpha) + start

    x2 = -r * np.sin(alpha) + start
    y2 = -r * np.cos(alpha) + start

    data = np.concatenate((np.array([x1, y1]), np.array([x2, y2])), axis=1)
    data = data.T
    labels = np.concatenate(
        (np.zeros(sample_num,
                  dtype=np.int32), np.ones(sample_num, dtype=np.int32)))
    labels = np.eye(2, dtype=np.int32)[labels]

    index = np.arange(2 * sample_num)
    np.random.shuffle(index)

    return data[index], labels[index]


# test part
if __name__ == "__main__":
    sample_num, circle_num, r_rate, offset, start = 100, 2, 0.4, 30, 0.5
    data, labels = generate_data(sample_num, circle_num, r_rate, offset, start)

    labels = np.argmax(labels, axis=1)
    spiral0 = data[labels == 0]
    spiral1 = data[labels == 1]
    print(spiral0.shape)
    plt.scatter(spiral0[:, 0], spiral0[:, 1])
    plt.scatter(spiral1[:, 0], spiral1[:, 1])
    plt.show()
