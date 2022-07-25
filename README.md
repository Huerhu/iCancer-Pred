# iCancer-Pred
A tool for identifying cancer and its type in the early stage

Web server: http://bioinfo.jcu.edu.cn/cancer or http://121.36.221.79/cancer


##  <a name="Data Download"></a> Data Download
Cancer datasets are downloaded at https://tcga.xenahubs.net. 
In the "TCGA Hub", DNA methylation (HumanMethylation450) data can be found and breast invasive carcinoma, colon adenocarcinoma, kidney renal clear cell carcinoma, kidney renal papillary cell carcinoma, liver hepatocellular carcinoma, lung adenocarcinoma and lung squamous cell carcinoma are chosen.

##  <a name="Data Processing"></a> Data Processing
### train_test_split.py
### del_nan.py
### CV_Elasticnet.py

##  <a name="Binary Classification"></a> Binary Classification
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
```python
overlap = np.load("elasticNet/XXXX_overlap.npy")
    train_feature = x[:, overlap]
```

##  <a name="Multi-Classification"></a> Multi-Classification
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
```python
overlap = np.load("elasticNet/XXXX_overlap.npy")
    train_feature = x[:, overlap]
```
##  <a name="Data Folder"></a> Data Folder
###  elasticNet folder :
"XXXX_overlap.npy" files are feature indexes obtained by using ElasticNet for feature selection in different cancer datasets.

###  site_name folder :
"XXXX_enet_site_name.csv" files are names of selected features, e.g."cg13332474".




