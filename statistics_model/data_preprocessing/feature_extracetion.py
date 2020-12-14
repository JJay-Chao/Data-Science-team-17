import pandas as pd
import tsfresh as tsf
import numpy as np
from sklearn.impute import KNNImputer
from statsmodels.tsa.seasonal import seasonal_decompose
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters

# Data
file_name = 'train_2'
data_path = r'C:\Users\USER\Desktop\DS\final\\' + file_name + ".csv"
train = pd.read_csv(data_path)
page = train['Page']
del train['Page']
mode = 'row_KNN'
size = train.shape[0]
# train = train.head(10)

complement_finish = True
feature_finish = False
if __name__ == '__main__':
    # complement missing value
    if not complement_finish:
        for index, row in train.iterrows():
            nan_num = row.isna().sum()
            if nan_num / row.size > 0.5:
                train = train.drop(index)
                page.drop(index)
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
        train['Page'] = page
        train.to_csv('./save_csv/' + file_name + "_complement.csv")
    else:
        train = pd.read_csv('./save_csv/' + file_name + "_complement.csv")
        del train['Page']
    # tsfresh feature extraction -> PCA -> clustering (k-means)
    # train["Page"] = page.head(10)
    if not feature_finish:
        feature_result = None
        for index, row in train.iterrows():
            temp = pd.DataFrame(row).T
            del temp['Unnamed: 0']
            temp = pd.DataFrame({'values': temp.T.values.squeeze(), 'times': temp.columns})
            temp['id'] = (page.loc[index])[0]
            settings = EfficientFCParameters()
            extracted_feature = tsf.extract_features(timeseries_container=temp, default_fc_parameters=settings, column_id="id", column_sort='times')
            extracted_feature['Page'] = page.loc[index]
            if feature_result is None:
                feature_result = extracted_feature
            else:
                feature_result = feature_result.append(extracted_feature)
        # print(extracted_feature)
        feature_result.to_csv('./save_csv/' + file_name + "_feature.csv", index=False)

    # cycle/seasonality, trend  extraction
    if not complement_finish:
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

    trend.to_csv('./save_csv/' + file_name + "_trend.csv", index=False)
    seasonal.to_csv('./save_csv/' + file_name + "_seasonal.csv", index=False)
    residual.to_csv('./save_csv/' + file_name + "_residual.csv", index=False)
