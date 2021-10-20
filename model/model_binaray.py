import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import tensorflow.keras.backend as K


def get_data():
    data = pd.read_csv("../data/train_test/XXXX/XXXX.csv")
    label = pd.read_csv("../data/train_test/XXXX/XXXX_label.csv")
    x = data.iloc[:, 1:].values
    y = label.iloc[:, 1].values.flatten()
    return x, y


def net_model(INPUT_SHAPE):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, input_shape=INPUT_SHAPE, activation='relu'),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def specific(confusion_matrix):
    '''recall = TP / (Tp + FN)'''
    specific = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    return specific


def train_model(x, y):
    overlap = np.load("../data/elasticNet/XXXX_overlap.npy")
    train_feature = x[:, overlap]
    skf = StratifiedKFold(shuffle=True, n_splits=5, random_state=0)
    true_labels = []
    probs = []
    predictions = []
    for train_index, valid_index in skf.split(train_feature, y):
        x_train, y_train = train_feature[train_index], y[train_index]
        x_valid, y_valid = train_feature[valid_index], y[valid_index]

        INPUT_SHAPE = x_train.shape
        model = net_model(INPUT_SHAPE)
        model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)
        prediction = model.predict(x_valid)
        prob = model.predict_proba(x_valid)
        true_labels = true_labels + y_valid.tolist()

        probs = probs + prob[:, 0].flatten().tolist()
        predictions = predictions + prediction.flatten().tolist()
        K.clear_session()
    predictions = np.array(predictions)
    predictions = predictions >= 0.5

    acc = accuracy_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    sp = specific(cm)
    sn = recall_score(true_labels, predictions, pos_label=1)
    mcc = matthews_corrcoef(true_labels, predictions)
    fpr_withoutAnn, tpr_withoutAnn, thresholds_withoutAnn = roc_curve(true_labels, probs, pos_label=1)
    AUC = auc(fpr_withoutAnn, tpr_withoutAnn)
    print(cm)
    print("acc: {:.5f} - std: {:.4f} ".format(np.mean(acc), np.std(acc)))
    print("s p: {:.5f} - std: {:.4f} ".format(np.mean(sp), np.std(sp)))
    print("s n: {:.5f} - std: {:.4f} ".format(np.mean(sn), np.std(sn)))
    print("mcc: {:.5f} - std: {:.4f} ".format(np.mean(mcc), np.std(mcc)))
    print("auc: {:.5f} - std: {:.4f} ".format(np.mean(AUC), np.std(AUC)))


if __name__ == '__main__':
    x, y = get_data()
    train_model(x, y)



