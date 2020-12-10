import pandas as pd
import tsfresh as tsf
import numpy as np
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import seasonal_decompose
from tsfresh.feature_extraction import ComprehensiveFCParameters

# Data
file_name = 'train_1'
data_path = r'C:\Users\USER\Desktop\DS\final\\' + file_name + ".csv"
train = pd.read_csv(data_path)
page = train['Page']
del train['Page']
mode = 'row_zero'
# train = train.head(10)
size = train.shape[0]

if __name__ == '__main__':
    # complement missing value
    for index, row in train.iterrows():
        nan_num = row.isna().sum()
        if nan_num / row.size > 0.5:
            train = train.drop(index)
        elif nan_num != 0:
            if mode == 'row_KNN':
                imputer = KNNImputer(n_neighbors=15, weights='uniform', metric='nan_euclidean')
                data = row.array
                data = np.expand_dims(data, axis=1)
                imputer.fit(data)
                data = imputer.transform(data)
                data = data.squeeze()
                row = pd.Series(data, index=row.index)
            elif mode == 'row_median':
                row = row.fillna(np.median(row))
            elif mode == 'row_mean':
                row = row.fillna(np.mean(row))
            elif mode == 'row_min':
                row = row.fillna(np.min(row))
            elif mode == 'row_max':
                row = row.fillna(np.max(row))
            elif mode == 'row_zero':
                row = row.fillna(0)
            train.loc[index] = row
        print("processing: {index}/{total} ".format(index=index, total=size))
    train.to_csv('./save_csv/' + file_name + "_complement.csv")

    # tsfresh feature extraction -> PCA -> clustering (k-means)
    # train["Page"] = page.head(10)
    train["Page"] = page
    y = train[train.columns[-2]]
    y.index = train['Page']
    settings = ComprehensiveFCParameters()
    extracted_feature = tsf.extract_relevant_features(y=y, timeseries_container=train, default_fc_parameters=settings,
                                                      column_id="Page")
    print(extracted_feature)
    extracted_feature.to_csv('./save_csv/' + file_name + "_feature.csv")

    # cycle/seasonality, trend  extraction
    del train['Page']
    trend = []
    seasonal = []
    residual = []
    for index, row in train.iterrows():
        decompose = seasonal_decompose(row, period=7)
        trend.append(decompose.trend)
        seasonal.append(decompose.seasonal)
        residual.append(decompose.resid)
        # print("seasonal: {seasonal}, trend: {trend}, residual: {resid}, observed: {observed}".format(
        #     seasonal=decompose.seasonal,
        #     trend=decompose.trend,
        #     resid=decompose.resid,
        #     observed=decompose.observed))
    trend = pd.DataFrame(trend)
    trend.columns = train.columns
    trend['Page'] = page

    seasonal = pd.DataFrame(seasonal)
    seasonal.columns = train.columns
    seasonal['Page'] = page

    residual = pd.DataFrame(residual)
    residual.columns = train.columns
    residual['Page'] = page

    trend.to_csv('./save_csv/' + file_name + "_trend.csv")
    seasonal.to_csv('./save_csv/' + file_name + "_seasonal.csv")
    residual.to_csv('./save_csv/' + file_name + "_residual.csv")
