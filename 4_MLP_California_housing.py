import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_california_housing
dataset = fetch_california_housing()

# 필요한 데이터 추출하기  dataset.속성, dataset[배열] 모두 가능
data = dataset.data
label = dataset.target
columns = dataset.feature_names

data = pd.DataFrame(data, columns=columns)

# 데이터 준비하기. 쪼개기
from sklearn.model_selection import train_test_split

# 학습용 데이터, 테스트용 데이터, 학습용 데이터 라벨, 테스트용 데이터 라벨
X_train,X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2023)

# 지도학습--> 멀티 레이어 퍼셉트론 회귀(Multi Layer Perceptron Regressor)
from sklearn.neural_network import MLPRegressor
mlp_regr = MLPRegressor(solver='adam', hidden_layer_sizes=200, max_iter=1000)
# mlp_regr에는 solver라는 파라미터를 설정한다. solver: lbfgs, sgd, adam 등이 있다.
# 여기까지 멀티 레이서 퍼셉트론 준비 완료 + max_iter 추가

mlp_regr.fit(X_train, y_train)
y_pred = mlp_regr.predict(X_test)

from sklearn.metrics import r2_score
print('다중 MLP 회귀, R2: {:.4f}'.format(r2_score(y_test, y_pred)))
# 다중 MLP 회귀, R2: 0.5996
# solver와 레이어 개수를 바꿔보며 R2값이 올라가는지 확인한다.
