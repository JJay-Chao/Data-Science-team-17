import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

file_name = 'train_2_feature'
data_path = r'./save_csv\\' + file_name + ".csv"
# data = pd.read_csv(data_path)
# page = data['Page']
# del data['Page']
# nan_columns = data.columns[data.isna().any()].tolist()
# for c in nan_columns:
#     del data[c]
# x = data.values
# x_scaled = scaler.fit_transform(x)
# data = pd.DataFrame(x_scaled)

# pca = KernelPCA(kernel='rbf', n_components=200)
# pca.fit(data)
# reduction_feature = pca.transform(data)
# reduction_feature_df = pd.DataFrame(reduction_feature)
# reduction_feature_df.to_csv("./save_csv\\" + file_name+"_reduction.csv", index=False)
count = {}
df = pd.read_csv("./save_csv\\" + file_name + "_clustering.csv")
for index, value in df['kmeans_30'].items():
    if str(value) not in count:
        count[str(value)] = 1
    else:
        count[str(value)] += 1
    print(index)
count = sorted(count.items(), key=lambda kv: kv[1])
print(count)

count1 = {}
df = pd.read_csv(r"C:\Users\USER\Downloads\feature_kmeans_suffix.csv")
for index, value in df['16.0'].items():
    if str(value) not in count1:
        count1[str(value)] = 1
    else:
        count1[str(value)] += 1
    print(index)
for key, value in count1.items():
    count1[key] = int((value * 34001) / 145061)
count1 = sorted(count1.items(), key=lambda kv: kv[1])
print(count, count.__len__())
print(count1, count1.__len__())

# kmeans = KMeans(n_clusters=30, random_state=42).fit(data)
# label = kmeans.predict(data)
# kmeans_df = pd.Series(label)
# kmeans_df['Page'] = page
# df = pd.read_csv("./save_csv\\" + file_name+"_clustering.csv")
# # print(df.columns)
# # df.columns = ['kmneas_10', 'Page']
# # df['kmeans_10'] = df['kmneas_10']
# # df = df[['Page', 'kmeans_5', 'kmeans_10']]
# df['kmeans_30'] = kmeans_df
# df.to_csv("./save_csv\\" + file_name+"_clustering.csv", index=False)
