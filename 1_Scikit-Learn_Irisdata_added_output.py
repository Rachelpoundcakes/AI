from sklearn.datasets import load_iris

# 변수 만들기 #
iris_dataset = load_iris()

## 데이터 내용 살펴보기 ##

# print(iris_datasets) # array 및 전체 내용 출력
# print(iris_dataset.DESCR) # description 확인하기

"""
참고
*배열로 가져오기; 배열과 속성의 프린트 결과는 똑같다.
-> iris_dataset['DESCR']
*속성으로 가져오기
-> iris_dataset.DESCR
"""

"""
print(iris_dataset) 결과

{'data': array([[5.1, 3.5, 1.4, 0.2],
       [4.9, 3. , 1.4, 0.2],
       [4.7, 3.2, 1.3, 0.2],
       [4.6, 3.1, 1.5, 0.2],
       ...
        [6.5, 3. , 5.2, 2. ],
       [6.2, 3.4, 5.4, 2.3],
       [5.9, 3. , 5.1, 1.8]]), 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 'frame': None, 'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'), 'DESCR':
하위에는 데이터 관련 설명 출력됨
...
conceptual clustering system finds 3 classes in the data.\n   
- Many, many more ...', 'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], 'filename': 'iris.csv', 'data_module': 'sklearn.datasets.data'}

print(iris_dataset.DESCR) 결과

**Data Set Characteristics:**

    :Number of Instances: 150 (50 in each of three classes)
    :Number of Attributes: 4 numeric, predictive attributes and the class
    :Attribute Information:
        - sepal length in cm
        - sepal width in cm
        - petal length in cm
        - petal width in cm
        - class:
                - Iris-Setosa
                - Iris-Versicolour
                - Iris-Virginica

    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    :Class Distribution: 33.3% for each of 3 classes.
    :Creator: R.A. Fisher
    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :Date: July, 1988
하위에는 데이터 관련 설명 출력됨
"""

print('iris_dataset의 key:\n', iris_dataset.keys())

"""
iris_dataset의 key:
 dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
"""

print('타깃의 이름:\n', iris_dataset['target_names'])
print('특징의 이름:\n', iris_dataset['feature_names'])

"""
타깃의 이름:
 ['setosa' 'versicolor' 'virginica']
특징의 이름:
 ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
"""

print('데이터의 타입:\n', type(iris_dataset['data']))
"""
*넘파이 배열
데이터의 타입:
 <class 'numpy.ndarray'>
"""

print('데이터의 크기:\n', iris_dataset['data'].shape)
"""
*shape은 데이터의 형태이다.
데이터의 크기:
 (150, 4)
"""

print('데이터의 처음 다섯 개:\n', iris_dataset['data'][:5])
"""
데이터의 처음 다섯 개:
 [[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
"""

print('타깃의 타입:\n', type(iris_dataset['target']))
"""
타깃의 타입:
 <class 'numpy.ndarray'>
"""

print('타깃의 크기:\n', iris_dataset['target'].shape)
"""
 (150,) -----> 1차원 결과
"""

print('타깃의 내용:\n', iris_dataset['target'])
"""
타깃의 내용:
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2] ----->  라벨링 되어 있는 데이터이다. 데이터를 가지고 시각화하거나 쪼갤 수 있다.
"""

### 데이터셋 준비하기 ###
"""
학습용(train), 테스트용(test) 데이터를 쪼개준다. 잘 섞어주어야 한다.
"""

from sklearn.model_selection import train_test_split

# 학습용 데이터, 테스트용 데이터, 학습용 데이터 라벨, 테스트용 데이터 라벨
# X는 대문자, y는 소문자로 쓰는 게 관행

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=2023)

print('X_train 형태: ', X_train.shape)
print('y_train 형태: ', y_train.shape)

print('X_test 형태: ', X_test.shape)
print('y_test 형태: ', y_test.shape)

"""
X_train 형태:  (112, 4)
y_train 형태:  (112,)
X_test 형태:  (38, 4)
y_test 형태:  (38,)
"""

### 여기까지 데이터 준비 끝 ###

#### 데이터 시각화하기
import pandas as pd
import matplotlib.pyplot as plt

# X_train 데이터
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])

print(iris_dataframe)
"""
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                  6.5               3.0                5.2               2.0
1                  5.4               3.4                1.7               0.2
2                  5.6               2.9                3.6               1.3
3                  6.3               2.9                5.6               1.8
4                  5.8               2.7                3.9               1.2
..                 ...               ...                ...               ...
107                6.9               3.1                4.9               1.5
108                6.5               3.0                5.5               1.8
109                4.6               3.1                1.5               0.2
110                5.0               3.0                1.6               0.2
111                6.3               2.3                4.4               1.3

[112 rows x 4 columns]
"""

# 데이터 예쁘게 만들기: scatter_matrix
pd.plotting.scatter_matrix(iris_dataframe, figsize=(8,8), 
                            marker='o', 
                            c=y_train, 
                            cmap='viridis', 
                            alpha=0.8)
plt.show()

##### 지도학습--> Classification: k-최근접 알고리즘(K-Nearest Neighbor, KNN을 사용하여 학습시키기(train) ####
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

# 학습시키기(train)
knn.fit(X_train, y_train)

###### ----->  학습 완료되어 knn에 학습 결과가 들어간다.

###### 2차원 이상의 넘파이 배열(임의의 수 샘플)을 하나 만들어,---> 예측하기(predict)
import numpy as np

X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new)
print(knn.predict(X_new))

###### -----> 예측 결과, array([0]), 즉 setosa로 나왔다.

# 예측 결과 보기 좋게 만들기
prediction = knn.predict(X_new)
print('예측:', prediction)
print('예측한 타깃의 이름: ', iris_dataset['target_names'][prediction])
"""
예측: [0]
예측한 타깃의 이름:  ['setosa']
"""

# 모델 평가하기
y_pred = knn.predict(X_test)

print('testset에 대한 예측값:\n', y_pred)
"""
testset에 대한 예측값:
 [2 1 1 2 1 2 1 1 0 1 0 1 0 2 0 2 0 1 0 0 1 0 2 1 0 0 0 2 1 0 0 0 0 1 1 2 0
 1]
"""

# y_pred(예측값)과 y_test(실측값)을 비교한다.
y_pred == y_test
np.mean(y_pred == y_test)
print('testset에 대한 정확도:\n', np.mean(y_pred==y_test)*100, '%')
"""
testset에 대한 정확도:
 97.36842105263158 %
"""

