# iCancer-Pred
A tool for identifying cancer and its type in the early stage

Web server: http://www.jci-bioinfo.cn/cancer is under maintenance.

You can visit this address: [121.36.221.79/cancer](121.36.221.79/cancer), it is available.

# Model Folder
-- "model_binaray.py" needs train/test datasets, train/test labels and indexes. 

e.g."BRCA_test_89.csv", "BRCA_test_label.csv" and "BRCA_overlap.npy"

-- "model_Multi.py" needs flies as same as "model_binaray.py", but only for "all_data".

# Data Folder 
-- elasticNet folder:
   "XXXX_overlap.npy" files are feature indexes obtained by using ElasticNet for feature selection in different cancer datasets.
 
-- site_name folder:
   "XXXX_enet_site_name.csv" files are names of selected features, e.g."cg13332474".

-- train_test folder:
   There are datasets and corresponding labels about training and testing in this paper.
