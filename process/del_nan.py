import pandas as pd
from tqdm import tqdm

"""
    XXXX: 二分类为BRCA/COAD/KIRC/KIRP/LIHC/LUAD/LUSC
          多分类为all_data
"""


def process_nan(file):
    count = 0
    data = pd.read_csv(file, chunksize=200)
    for chunksize in tqdm(data):
        columns_name = chunksize.columns
        df = chunksize.dropna(how='any', axis=0, subset=columns_name[1:])
        if count == 0:
            df.to_csv("XXXX_train_delnan.csv", mode="a", index=False)
            count = 1
        else:
            df.to_csv("XXXX_train_delnan.csv", mode="a", index=False, header=None)


if __name__ == '__main__':
    file_path = 'XXXX_train_data.csv'
    process_nan(file_path)

