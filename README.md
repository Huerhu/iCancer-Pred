# iCancer-Pred
A tool for identifying cancer and its type in the early stage

Web server: http://bioinfo.jcu.edu.cn/cancer or http://121.36.221.79/cancer


##  <a name="Data Download"></a> Data Download
Cancer datasets are downloaded at https://tcga.xenahubs.net. 
In the "TCGA Hub", DNA methylation (HumanMethylation450) data can be found and breast invasive carcinoma, colon adenocarcinoma, kidney renal clear cell carcinoma, kidney renal papillary cell carcinoma, liver hepatocellular carcinoma, lung adenocarcinoma and lung squamous cell carcinoma are chosen.

##  <a name="Data Processing"></a> Data Processing
##  <a name="Binary Classification"></a> Binary Classification
##  <a name="Multi-Classification"></a> Multi-Classification
##  <a name="Data Folder"></a> Data Folder

###  elasticNet folder :
"XXXX_overlap.npy" files are feature indexes obtained by using ElasticNet for feature selection in different cancer datasets.

###  site_name folder :
"XXXX_enet_site_name.csv" files are names of selected features, e.g."cg13332474".




