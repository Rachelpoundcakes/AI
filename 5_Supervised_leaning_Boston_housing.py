#Scikit Learn에서 사용하는 Supervised Learning을 해 보자

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

#warning 메시지 무시하기(불필요한 warning 방지)
import warnings
warnings.filterwarnings('ignore')

#보스턴 집값 데이터 불러오기
from sklearn.datasets import load_boston
boston = load_boston()

print(boston['DESCR'])

#필요한 데이터 추출하기
#data = boston['data']로 써도 되지만 아래를 선호
data = boston.data
label = boston.target
columns = boston.feature_names

data = pd.DataFrame(data, columns=columns)
data.head()

data.shape

#Simple Linear Regression을 해 보자.
#데이터 준비하기. 쪼개기
from sklearn.model_selection import train_test_split

#학습용 데이터, 테스트용 데이터, 학습용 데이터 라벨, 테스트용 데이터 라벨
X_train, X_test, y_train, y_test= train_test_split(data, label, test_size=0.2, random_state=2022)

print(X_train['RM'])

#1차원 배열 -> 2차원 배열로 바꿔주기
#(-1, 1)로!!
#[:5]스압 있으니 5개만 보자!
X_train['RM'].values.reshape(-1, 1)[:5]

from sklearn.linear_model import LinearRegression
sim_lr = LinearRegression()

#데이터, 라벨 주기
#RM 보스턴 집 방의 개수
sim_lr.fit(X_train['RM'].values.reshape((-1,1)) ,y_train)

#예측치. 룸의 개수로 예측
y_pred = sim_lr.predict(X_test['RM'].values.reshape((-1,1)))

print(y_pred)

#결과 살펴보기
#결정계수로 학습한 것을 확인. 원본데이터와 예측값 차이 비교하기
from sklearn.metrics import r2_score
print('단순 선형 회귀, R2: {:.4f}'.format(r2_score(y_test, y_pred)))
#.4f ===> 실수, 소수점 넷째자리까지

#결과 시각화 x, y축
#legend 각주
line_X = np.linspace(np.min(X_test['RM']), np.max(X_test['RM']), 10)
line_y = sim_lr.predict(line_X.reshape(-1,1))

plt.scatter(X_test['RM'], y_test, s=10, c='black')
plt.plot(line_X, line_y, c='red')
plt.legend(['Regression line','Test data sample'], loc='upper left')

#너무 분산되어 나옴. 실제 값과 예측값의 차이多. Room 하나만 가지고 평가하기 부족.

#Multiple Linear Regression
mul_lr = LinearRegression()

#RM column 말고 전체 columns를 사용
mul_lr.fit(X_train, y_train)

y_pred = mul_lr.predict(X_test)

#y 테스트용 데이터, 예측치
print('다중 선형 회귀, R2: {:.4f}'.format(r2_score(y_test, y_pred)))

#아까보다 3배 정도 향상된 모델 But 못 씀... 정확도가 너무 떨어짐

#Decision Tree Regressor 결정 트리 모델
from sklearn.tree import DecisionTreeRegressor
dt_regr =DecisionTreeRegressor(max_depth=2)

#2차원 배열로 바꾸기. 학습시키기
dt_regr.fit(X_train['RM'].values.reshape((-1,1)), y_train)

#예측치
y_pred = dt_regr.predict(X_test['RM'].values.reshape(-1,1))

print('단순 결정 트리 회귀 R2: {:.4f}'.format(r2_score(y_test, y_pred)))

#32.47%

#max_depth 변화를 줘 본다. 너무 높이면 학습 데이터와 오버피팅된다. 테스트용 데이터와 격차가 벌어짐
#반복문을 써서 가장 좋은 depth를 쓴다. 아래에 만들어 보자

#배열 만들기
arr = np.arange(1,11)
print(arr)

best_depth = 1
best_r2 = 0
for depth in arr:
  dt_regr = DecisionTreeRegressor(max_depth=depth)
  dt_regr.fit(X_train['RM'].values.reshape((-1,1)), y_train)
  y_pred = dt_regr.predict(X_test['RM'].values.reshape(-1,1))
  
  temp_r2 = r2_score(y_test, y_pred)
  print('\n단순 결정 트리 회귀 depth={} R2: {:.4f}'.format(depth, temp_r2))

  if best_r2 < temp_r2:
    best_depth = depth
    best_r2 = temp_r2

print('최적의 결과는 depth={} r2={:.4f}'.format(best_depth, best_r2))

dt_regr = DecisionTreeRegressor(max_depth=8)
dt_regr.fit(X_train, y_train)

y_pred = dt_regr.predict(X_test)
print('다중 결정 트리 R2: {:.4f}'.format(r2_score(y_test, y_pred)))

#Support Vector Machine Regressor
#알고리즘 불러오기
from sklearn.svm import SVR
svm_regr = SVR(C=1)

svm_regr.fit(X_train['RM'].values.reshape(-1,1), y_train)
y_pred = svm_regr.predict(X_test['RM'].values.reshape(-1,1))

print('단순 서포트 벡터 머신 회귀 R2: {:.4f}'.format(r2_score(y_test,y_pred)))

#결과의 시각화 X, y 축
line_X = np.linspace(np.min(X_test['RM']), np.max(X_test['RM']), 100)
line_y = svm_regr.predict(line_X.reshape(-1,1))

plt.scatter(X_test['RM'], y_test, c='black')
plt.plot(line_X, line_y, c='red')
plt.legend(['Regression line', 'Test data sample'], loc='upper left')

#C=20 일반화
svm_regr = SVR(C=20)
svm_regr.fit(X_train, y_train)
y_pred = svm_regr.predict(X_test)
print('다중 서포트 벡터 머신 회귀, R2 : {:.4f}'.format(r2_score(y_test, y_pred)))

arr = np.arange(10000,25000)
arr

best_C = 0
best_r2 = 0


for C in arr:
  svm_regr = SVR(C=C)
  svm_regr.fit(X_train, y_train)
  y_pred = svm_regr.predict(X_test)
  #print('다중 서포트 벡터 머신 회귀, R2 : {:.4f}'.format(r2_score(y_test, y_pred)))

  if best_r2 < temp_r2:
    best_depth = depth
    best_r2 = temp_r2

print('최적의 결과는 depth={} r2={:.4f}'.format(best_depth, best_r2))

#Multi Layer Perceptron Regressor
from sklearn.neural_network import MLPRegressor
mlp_regr = MLPRegressor(solver='adam',hidden_layer_sizes=100) #solver: lbfgs, sgd, adam 등이 있다.
#여기까지 멀티 레이서 퍼셉트론 준비 완료

mlp_regr.fit(X_train, y_train)
y_pred = mlp_regr.predict(X_test)

print('다중 MLP 회귀, R2: {:.4f}'.format(r2_score(y_test,y_pred)))

#몇 번 돌리는지에 따라 결과는 조금씩 달라진다.

#max_iter 추가
from sklearn.neural_network import MLPRegressor
mlp_regr = MLPRegressor(solver='adam',hidden_layer_sizes=100,max_iter=1000) #solver: lbfgs, sgd, adam 등이 있다.
#여기까지 멀티 레이어 퍼셉트론 준비 완료

mlp_regr.fit(X_train, y_train)
y_pred = mlp_regr.predict(X_test)

print('다중 MLP 회귀, R2: {:.4f}'.format(r2_score(y_test,y_pred)))
