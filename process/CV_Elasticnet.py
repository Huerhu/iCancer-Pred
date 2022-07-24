import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats
from sklearn.linear_model import ElasticNetCV


def get_data():
    data = pd.read_csv("XXXX_train_delnan.csv")
    label = pd.read_csv("XXXX_train_label.csv")
    df_data = data.iloc[:, 1:].T.values
    df_label = label.iloc[:, 1].values.flatten()
    return df_data, df_label,


def select_feature(data, label):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    i = 1
    for train_index, test_index in skf.split(data, label):
        x_train, y_train = data[train_index], label[train_index]  # shape() 前者为样本数，后者为位点数

        cv_train = x_train.T
        var = stats.variation(cv_train, axis=1)
        index1 = np.where(var > 1)
        np.save("CV/XXXX" + str(i) + "次折叠获得的特征" + ".npy", index1[0])

        index = index1[0]

        elanet_train = x_train[:, index]
        enetCV = ElasticNetCV(alphas=[0.0001], l1_ratio=[.1], max_iter=5000).fit(elanet_train, y_train)
        mask = enetCV.coef_ != 0
        index2 = np.where(mask == True)  # True的下标，feature_index是个元组
        # 保存下标，feature_index[0]是个列表
        np.save("elasticNet/XXXX第" + str(i) + "次折叠获得的特征" + ".npy", index2[0])

        i = i + 1


def feature_process_cv():
    data1 = np.load("CV/XXXX第1次折叠获得的特征.npy")
    data2 = np.load("CV/XXXX第2次折叠获得的特征.npy")
    data3 = np.load("CV/XXXX第3次折叠获得的特征.npy")
    data4 = np.load("CV/XXXX第4次折叠获得的特征.npy")
    data5 = np.load("CV/XXXX第5次折叠获得的特征.npy")
    # 取重复部分
    feature1 = np.intersect1d(data1, data2)
    feature2 = np.intersect1d(feature1, data3)
    feature3 = np.intersect1d(feature2, data4)
    overlap1 = np.intersect1d(feature3, data5)
    np.save("CV/XXXX五次折叠后特征的交集（重复）.npy", overlap1)


def feature_process_elasticNet():
    data1 = np.load("elasticNet/XXXX第1次折叠获得的特征.npy")
    data2 = np.load("elasticNet/XXXX第2次折叠获得的特征.npy")
    data3 = np.load("elasticNet/XXXX第3次折叠获得的特征.npy")
    data4 = np.load("elasticNet/XXXX第4次折叠获得的特征.npy")
    data5 = np.load("elasticNet/XXXX第5次折叠获得的特征.npy")
    # 取重复部分
    feature1 = np.intersect1d(data1, data2)
    feature2 = np.intersect1d(feature1, data3)
    feature3 = np.intersect1d(feature2, data4)
    overlap2 = np.intersect1d(feature3, data5)
    np.save("elasticNet/XXXX五次折叠后特征的交集（重复）.npy", overlap2)


if __name__ == '__main__':
    data, label = get_data()
    select_feature(data, label)
    feature_process_cv()
    feature_process_elasticNet()
