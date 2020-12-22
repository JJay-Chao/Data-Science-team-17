# Data Science Final Proj. Feature Extraction 

## TODO & Done
- feature extraction using tsfresh 
    - [x] feature extraction
    - [x] additive trend, seasonality, residual extraction
    - [ ] multiplicative trend, seasonality, residual extraction
    - intimate feature
        - [x]  since the `tsfresh` took long time to process, the handmade feature will be tried to imitate the distribution of extracted feature. 
    - dimension reduction and clustering
        - dimension reduction 
            - [x] PCA
            - [ ] Autoencoder
            - [ ] LDA
        - clustering
            - [x] k means
            - [ ] hierarchical clustering
     - complement missing value method 
        - [ ] each model will have their own imputing method  
           
## Data
-  train 1 complement
-  train 1 trend
-  train 1 seasonality
-  train 1 residual
-  train 1 feature (sample)
-  train 2 complement
-  train 2 trend
-  train 2 seasonality
-  train 2 residual
> 2020.12.23 update: 
-  train 2 feature
-  train 2 feature_handcraft 

## Data Format
- first column might be index, you can ignore it.
- following columns will be value for each time series and days.
- last column will be Page

## Seasonal Decompose
- Using **additive** model: which means 
 ` result = Trend + Seasonal + Residual`, and the period is set to `7` temporarily 

## Complement Missing Value
- using KNN to impute missing value temporarily
- time series will be removed if missing value is higher than **50%** 

## Feature Extraction(W.I.P)
- `train1_feature.csv` is a sample feature extraction file, you can refer to the columns name to know what features will be included. 
> 2020.12.23 update
- `train2_feature.csv` is feature extraction file using `tsfresh`, only contain **36000** row temporarily(W.I.P)
- `train2_feature_handcraft.csv` is feature extraction file by extracting 9 common statistic, contain **138432** row(same as `train2_complement.csv`). 
## Usage
download data from [here](https://drive.google.com/drive/folders/1FuRki8KuII1hj-868KJeClXOgTqKcYVj?usp=sharing)  

 