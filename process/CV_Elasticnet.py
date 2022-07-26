import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats
from sklearn.linear_model import ElasticNetCV

"""
    Binary Classification
        XXXX: BRCA/COAD/KIRC/KIRP/LIHC/LUAD/LUSC
        
    Multi-Classification
        XXXX: all_data
"""


def select_feature(data, label, i):
    cv_train = data.T
    var = stats.variation(cv_train, axis=1)
    index1 = np.where(var > 1)
    np.save("CV/XXXX第" + str(i) + "次折叠获得的特征" + ".npy", index1[0])

    index_1 = index1[0]

    elanet_train = data[:, index_1]
    enetCV = ElasticNetCV(alphas=[0.0001], l1_ratio=[.1], max_iter=5000).fit(elanet_train, label)
    mask = enetCV.coef_ != 0
    index2 = np.where(mask == True)
    np.save("elasticNet/XXXX第" + str(i) + "次折叠获得的特征" + ".npy", index2[0])

    index_2 = index2[0]

    return index_2


def feature_process_cv():
    data1 = np.load("CV/XXXX第1次折叠获得的特征.npy")
    data2 = np.load("CV/XXXX第2次折叠获得的特征.npy")
    data3 = np.load("CV/XXXX第3次折叠获得的特征.npy")
    data4 = np.load("CV/XXXX第4次折叠获得的特征.npy")
    data5 = np.load("CV/XXXX第5次折叠获得的特征.npy")
    feature1 = np.intersect1d(data1, data2)
    feature2 = np.intersect1d(feature1, data3)
    feature3 = np.intersect1d(feature2, data4)
    overlap1 = np.intersect1d(feature3, data5)
    np.save("CV/XXXX_overlap.npy", overlap1)


def feature_process_elasticNet():
    data1 = np.load("elasticNet/XXXX第1次折叠获得的特征.npy")
    data2 = np.load("elasticNet/XXXX第2次折叠获得的特征.npy")
    data3 = np.load("elasticNet/XXXX第3次折叠获得的特征.npy")
    data4 = np.load("elasticNet/XXXX第4次折叠获得的特征.npy")
    data5 = np.load("elasticNet/XXXX第5次折叠获得的特征.npy")
    feature1 = np.intersect1d(data1, data2)
    feature2 = np.intersect1d(feature1, data3)
    feature3 = np.intersect1d(feature2, data4)
    overlap2 = np.intersect1d(feature3, data5)
    np.save("elasticNet/XXXX_overlap.npy", overlap2)


if __name__ == '__main__':
    feature_process_cv()
    feature_process_elasticNet()
