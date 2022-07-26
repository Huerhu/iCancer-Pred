# iCancer-Pred
A tool for identifying cancer and its type in the early stage

Web server: http://bioinfo.jcu.edu.cn/cancer or http://121.36.221.79/cancer


##  <a name="Data Download"></a> Data Download
Cancer datasets are downloaded at https://tcga.xenahubs.net. 
In the "TCGA Hub", DNA methylation (HumanMethylation450) data can be found and breast invasive carcinoma, colon adenocarcinoma, kidney renal clear cell carcinoma, kidney renal papillary cell carcinoma, liver hepatocellular carcinoma, lung adenocarcinoma and lung squamous cell carcinoma are chosen.

##  <a name="Data Processing"></a> Data Processing
### train_test_split.py
For each cancer, HumanMethylation450 data are randomly split into a training dataset (90%) and an independent testing dataset (10%).
```python
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.1, random_state=42, shuffle=True)
```
### del_nan.py
In the training datasets, some sites which include "Nan" value are removed.
```python
df = chunksize.dropna(how='any', axis=0, subset=columns_name[1:])
```
### CV_Elasticnet.py
For the training datasets, after removing "Nan" value, dimensionality reduction and feature selection are achieved by thresholding the coefficient of variation and employing elastic network.
```python
var = stats.variation(cv_train, axis=1)
index1 = np.where(var > 1)
```
```python
enetCV = ElasticNetCV(alphas=[0.0001], l1_ratio=[.1], max_iter=5000).fit(elanet_train, y_train)
mask = enetCV.coef_ != 0
index2 = np.where(mask == True)
```
##  <a name="Binary Classification"></a> Binary Classification
For binary classification, the fully connected network structure is as follows:
```python
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
```
The features selected by using ElasticNet are used as the input of the classifier.
```python
overlap = np.load("elasticNet/XXXX_overlap.npy")
train_feature = x_train[:, overlap]
```

##  <a name="Multi-Classification"></a> Multi-Classification
For multi-classification, the structure of the fully connected network is as follows:
```python
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
```
##  <a name="Data Folder"></a> Data Folder
###  elasticNet folder :
"XXXX_overlap.npy" files are feature indexes obtained by using ElasticNet for feature selection in different cancer datasets.
###  site_name folder :
"XXXX_enet_site_name.csv" files are names of selected features, e.g."cg13332474".




