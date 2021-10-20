import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from focal_loss import multi_category_focal_loss2


def get_data():
    data = pd.read_csv("../data/train_test/all_data/XXXX.csv")
    label = pd.read_csv("../data/train_test/all_data/XXXX_label.csv")
    x = data.iloc[:, 1:].values
    y = label.iloc[:, 1].values.flatten()
    return x, y


def net_model(INPUT_SHAPE):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=INPUT_SHAPE, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    model.compile(loss=multi_category_focal_loss2(alpha=0.25, gamma=1), optimizer='adam', metrics=['accuracy'])
    return model


def to_onehot(label):
    onehot_label = np.zeros((len(label), 8))
    for i in range(len(label)):
        onehot_label[i, label[i]] = 1
    return onehot_label


def train_model(x, y):
    overlap = np.load("../data/elasticNet/all_data_overlap.npy")
    train_feature = x[:, overlap]

    skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    accuracys = []
    precisions = []
    recalls = []
    mccs = []
    kappas = []
    true_label = []
    prediction_label = []
    for train_index, valid_index in skf.split(train_feature, y):
        x_train, y_train = train_feature[train_index], y[train_index]
        x_valid, y_valid = train_feature[valid_index], y[valid_index]

        INPUT_SHAPE = x_train.shape

        y_train = to_onehot(y_train)
        y_valid = to_onehot(y_valid)
        model = net_model(INPUT_SHAPE)
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=30, batch_size=32, verbose=1)

        y_valid = np.argmax(y_valid, axis=1)
        y_pre = model.predict_classes(x_valid).flatten()

        accuracy = accuracy_score(y_valid, y_pre)
        accuracys.append(accuracy)
        precision = precision_score(y_valid, y_pre, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro')
        precisions.append(precision)
        recall = recall_score(y_valid, y_pre, labels=[0, 1, 2, 3, 4, 5, 6, 7], average='macro')
        recalls.append(recall)
        mcc = matthews_corrcoef(y_valid, y_pre)
        mccs.append(mcc)
        kappa = cohen_kappa_score(y_valid, y_pre, labels=None)
        kappas.append(kappa)
        true_label += y_valid.tolist()
        prediction_label += y_pre.tolist()
        cm = confusion_matrix(true_label, prediction_label)
        print(cm)
        print("accuracy = ", accuracy)
        print("precision = ", precision)
        print("recall = ", recall)
        print("mcc = ", mcc)
        print("kappa = ", kappa)
    print("平均accuracy = {:.3%} -std: {:.3f} ".format(np.mean(accuracys), np.std(accuracys)))
    print("平均precision = {:.3%} -std: {:.3f} ".format(np.mean(precisions), np.std(precisions)))
    print("平均recall = {:.3%} -std: {:.3f} ".format(np.mean(recalls), np.std(recalls)))
    print("平均mcc = {:.3%} -std: {:.3f} ".format(np.mean(mccs), np.std(mccs)))
    print("平均kappa = {:.3%} -std: {:.3f} ".format(np.mean(kappas), np.std(kappas)))


if __name__ == '__main__':
    x, y = get_data()
    train_model(x, y)

