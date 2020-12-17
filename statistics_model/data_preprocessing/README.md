# Data Science Final Proj. Feature Extraction 

## TODO & Done
- feature extraction using tsfresh 
    - [ ] feature extraction
    - [x] additive trend, seasonality, residual extraction
    - [ ] multiplicative trend, seasonality, residual extraction
    - intimate feature
        - [ ]  since the `tsfresh` took long time to process, the handmade feature will be tried to imitate the distribution of extracted feature. 
    - dimension reduction and clustering
        - dimension reduction 
            - [ ] PCA
            - [ ] Autoencoder
            - [ ] LDA
        - clustering
            - [ ] k means
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

## Usage
download data from [here](https://drive.google.com/drive/folders/1FuRki8KuII1hj-868KJeClXOgTqKcYVj?usp=sharing)  

 