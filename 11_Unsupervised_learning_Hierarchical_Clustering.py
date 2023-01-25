import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_wine
wine = load_wine()

data = wine.data
label = wine.target
columns = wine.feature_names

data = pd.DataFrame(data, columns=columns)

# 데이터 전처리 필요 -> MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

print(data.shape)
# (178, 13)

# 13차원에서는 패턴을 찾기 힘드므로 차원을 축소한다
# PCA(차원의 축소)
# PCA (차원의 축소)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(data)
print(data)
"""
[[-7.06335756e-01 -2.53192753e-01]
 [-4.84976802e-01 -8.82289142e-03]
 [-5.21172266e-01 -1.89187222e-01]
 ...
 [ 5.72991102e-01 -4.25516087e-01]  
 [ 7.01763997e-01 -5.13504983e-01]]

==> 13차원의 데이터가 2차원으로 변환되었다. 전처리 끝.
"""

# Unsupervised learning: Hierarchical Clustering

from sklearn.cluster import AgglomerativeClustering
single_clustering = AgglomerativeClustering(n_clusters=3,linkage='single')
complete_clustering = AgglomerativeClustering(n_clusters=3,linkage='complete')
average_clustering = AgglomerativeClustering(n_clusters=3,linkage='average')

# 학습시키기
single_clustering.fit(data)
complete_clustering.fit(data)
average_clustering.fit(data)

single_cluster = single_clustering.labels_
complete_cluster = complete_clustering.labels_
average_cluster = average_clustering.labels_

# 세 개 모두 프린트해서 결과를 보고 하나를 고른다.
print(single_cluster)
print(complete_cluster)
print(average_cluster)

"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
[1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 0 1 1 1 1 0 1 0 0 0
 0 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0
 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

 ==> 시각화해서 비교해보자.
"""
plt.scatter(data[:,0], data[:,1],c=single_cluster)
plt.show()
plt.scatter(data[:,0], data[:,1],c=complete_cluster)
plt.show()
plt.scatter(data[:,0], data[:,1],c=average_cluster)
plt.show()
plt.scatter(data[:,0], data[:,1],c=label)
plt.show()

# Dendrogram

from scipy.cluster.hierarchy import dendrogram
plt.figure(figsize=(10,10))
children = single_clustering.children_
distance = np.arange(children.shape[0])
no_of_observations = np.arange(2, children.shape[0]+2)
linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

dendrogram(linkage_matrix, p=len(data), labels=single_cluster,
           show_contracted=True, no_labels=True)

# Silhouette

# Silhouette

from sklearn.metrics import silhouette_score

best_n = 1
best_score = -1

for n_cluster in range(2,11):
  average_clustering = AgglomerativeClustering(n_clusters=n_cluster,linkage='average')
  average_clustering.fit(data)
  cluster = average_clustering.labels_
  score = silhouette_score(data, cluster)

  print('클러스터의 수: {} 실루엣 점수:{:.2f}'.format(n_cluster, score))

  if score > best_score:
    best_n = n_cluster
    best_score = score

print('가장 높은 실루엣 점수를 가진 클러스터 수: {}, 실루엣 점수 {:.2f}'.format(best_n, best_score))
"""
클러스터의 수: 2 실루엣 점수:0.49
클러스터의 수: 3 실루엣 점수:0.56
클러스터의 수: 4 실루엣 점수:0.48
클러스터의 수: 5 실루엣 점수:0.42
클러스터의 수: 6 실루엣 점수:0.37
클러스터의 수: 7 실루엣 점수:0.34
클러스터의 수: 8 실루엣 점수:0.34
클러스터의 수: 9 실루엣 점수:0.37
클러스터의 수: 10 실루엣 점수:0.33
가장 높은 실루엣 점수를 가진 클러스터 수: 3, 실루엣 점수 0.56
"""