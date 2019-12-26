'''
@Author: Da Ma
@Date: 2019-12-23 03:25:07
@LastEditTime : 2019-12-26 10:01:33
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Proj/src/utils/Plot.py
'''

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    """to plot the decision boundary for the classifier.

    Args:
        clf (model): the classifier
        X (numpy): the data of the samples
        Y (numpy): the labels of the samples
        cmap (str, optional): the color design. Defaults to 'Paired_r'.
    """
    h = 0.001
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 19 * h, X[:, 1].max() + 10 * h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).asnumpy()
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, edgecolors='k')
    plt.show()
