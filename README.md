# iCancer-Pred
A tool for identifying cancer and its type in the early stage

Web server: http://bioinfo.jcu.edu.cn/cancer or http://121.36.221.79/cancer


##  <a name="Data Download"></a> Data Download




# Model Folder
-- "model_binaray.py" needs train/test datasets, train/test labels and indexes. 

e.g."BRCA_test_89.csv", "BRCA_test_label.csv" and "BRCA_overlap.npy"

-- "model_Multi.py" needs flies as same as "model_binaray.py", but only for "all_data".

# Data Folder 
-- elasticNet folder:
   "XXXX_overlap.npy" files are feature indexes obtained by using ElasticNet for feature selection in different cancer datasets.
 
-- site_name folder:
   "XXXX_enet_site_name.csv" files are names of selected features, e.g."cg13332474".


