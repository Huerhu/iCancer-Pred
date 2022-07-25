import pandas as pd
from sklearn.model_selection import train_test_split

"""
    XXXX: 
        Binary Classification: BRCA/COAD/KIRC/KIRP/LIHC/LUAD/LUSC
        Multi-Classification: all_data
"""


def get_data():
    methy_data = pd.read_csv(r'HumanMethylation450', sep='\t')
    sample_name = methy_data.columns
    CG_id = methy_data.iloc[:, 0].tolist()
    df_data = methy_data.iloc[:, 1:].T
    return sample_name, CG_id, df_data


def get_label(sample_name):
    label_data = pd.DataFrame(index=["label"], columns=sample_name[1:])
    labels = []
    for i in sample_name[1:]:
        if i[-2:] == '11':
            labels.append('0')
        else:
            labels.append('1')
            # For multi-Classification, BRCA:'1', COAD:'2', KIRC:'3', KIRP:'4', LIHC:'5', LUAD:'6', LUSC:'7'
    label_data.loc['label'] = labels
    label_data.to_csv('XXXX_label.csv', index=False)
    df_label = label_data.T
    return df_label


def train_test(data, label, name):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42, shuffle=True)
    x_train.columns = name
    x_test.columns = name
    y_train.columns = ['label']
    y_test.columns = ['label']
    x_train = x_train.T
    x_test = x_test.T
    x_train.to_csv("XXXX_train_data.csv")
    x_test.to_csv("XXXX_valid_data.csv")
    y_train.to_csv("XXXX_train_label.csv")
    y_test.to_csv("XXXX_valid_label.csv")
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    samples_name, CG_ids, data = get_data()
    label = get_label(samples_name)
    train_test(data, label, CG_ids)
