from sklearn.datasets import load_iris
#변수 만들기
iris_dataset = load_iris()

#description 확인하기
print(iris_dataset.DESCR)

print('iris_dataset의 키:\n', iris_dataset.keys())

#라벨 데이터는 타켓 항목에, DESCR에는 설명이 들어있다.

#배열로 가져오기(결과는 똑같다)
iris_dataset['DESCR']
#속성으로 가져오기
iris_dataset.DESCR

print('타깃의 이름: \n', iris_dataset['target_names'])

print('특징의 이름: \n', iris_dataset['feature_names'])

#여기에 있는 데이터는 넘파이 배열로 되어 있다.
print('data의 타입:', type(iris_dataset['data']))

print('data의 크기', iris_dataset['data'].shape)
#shape= 데이터의 구조

print('data의 처음 다섯 개:\n', iris_dataset['data'][:5])
#[:5] 처음부터 5개까지

print('target의 타입', type(iris_dataset['target']))
#NumPy 배열

print('target의 크기', iris_dataset['target'].shape)

#target의 크기 (150,) -----> 1차원 결과

print('target의 내용', 'target:\n', iris_dataset['target'])

#라벨링 되어 있는 데이터이다. 이 데이터를 가지고 시각화하거나 쪼갤 수 있다.

#데이터의 준비.
#학습용, 테스트용 데이터를 쪼개준다. 잘 섞어 주어야 한다!!
from sklearn.model_selection import train_test_split

train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)

#학습용 데이터, 테스트용 데이터, 라벨, 라벨
#X는 대문자, y소문자로 쓰는 게 관행
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)

print('X_train 크기:', X_train.shape)
print('y_train 크기:', y_train.shape)

print('X_test 크기:', X_test.shape)
print('y_test 크기:', y_test.shape)

#test_size=0.5) 추가
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0, test_size=0.3)

print('X_train 크기:', X_train.shape)
print('y_train 크기:', y_train.shape)

print('X_test 크기:', X_test.shape)
print('y_test 크기:', y_test.shape)

##여기까지 데이터 준비 끝

#데이터의 시각화
import pandas as pd
import matplotlib .pyplot as plt

#X_train = 데이터
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])

#데이터 예쁘게 만들기 = scatter_matrix
pd.plotting.scatter_matrix(iris_dataframe)

#더 보기 좋게 만들기
pd.plotting.scatter_matrix(iris_dataframe, figsize=(15,15), 
                           marker='o', 
                           c=y_train, 
                           cmap='viridis',
                           alpha=0.8)

# 첫 번째 머신 러닝 모델: k-최근접 알고리즘
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

#학습시키기
knn.fit(X_train, y_train)

#학습 완료되어 knn에 학습 결과가 들어감

#넘파이 배열 하나 만든 다음(임의의 수 샘플), ---> 예측하기
import numpy as np

#최소한 2차원 데이터를 줘야 한다
X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new)

knn.predict(X_new)

#예측 결과 array([0]), 즉 setosa로 나옴!!

#보기 좋게 만들기
prediction = knn.predict(X_new)
print('예측:', prediction)
print('예측한 타깃의 이름:', iris_dataset['target_names'][prediction])

#결과---> 예측: [0],  예측한 타깃의 이름: ['setosa']

#잘 만들었을까? 모델 평가하기
y_pred = knn.predict(X_test)

print('테스트 세트에 대한 예측값:\n', y_pred)

#y_pred(예측값)과 y_test(실측값)을 비교한다.
y_pred == y_test

#False 하나 제외하고 모두 True가 나왔으므로 높은 정확도를 보임

np.mean(y_pred == y_test)

#0.97이므로 97%의 정확도를 가진 모델이라고 말할 수 있다.

print('테스트 세트에 대한 정확도:\n', np.mean(y_pred==y_test)*100, '%')

#평균치 말고 스코어로 정확도를 낼 수도 있다.
print('테스트 세트에 대한 정확도:\n', knn.score(X_test, y_test), '%')

