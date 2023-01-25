import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn # Machine Learning Framework 
import os
from os.path import join

abalone_path = join('.', 'abalone.txt') #.은 현재경로 ..는 상위 경로
column_path = join('.', 'abalone_attributes.txt')

print(abalone_path)  # 암수 정보 데이터. 어떤 조건이면 암컷이 되고, 수컷이 되는지 M(수컷), F(암컷), I(미정)

abalone_columns = list() # 리스트 객체로 만들기
for line in open(column_path):
  abalone_columns.append(line)

print(abalone_columns)
"""
['Sex\n', 'Length\n', 'Diameter\n', 'Height\n', 'Whole weight\n', 'Shucked weight\n', 'Viscera weight\n', 'Shell weight\n', 'Rings']
"""

# 불필요한 요소들 잘라내기. 위의 \n처럼 같이 출력되는 것들
abalone_columns = list()
for line in open(column_path):
  abalone_columns.append(line.strip()) #abalone_columns에 저장하기

print(abalone_columns)
"""
['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
"""

pd.read_csv(abalone_path, header = None, names = abalone_columns)

# data 변수에 저장
data = pd.read_csv(abalone_path, header = None, names = abalone_columns)

print(data)
"""
     Sex  Length  Diameter  Height  Whole weight  Shucked weight  Viscera weight  Shell weight  Rings
0      M   0.455     0.365   0.095        0.5140          0.2245          0.1010        0.1500     15
1      M   0.350     0.265   0.090        0.2255          0.0995          0.0485        0.0700      7
2      F   0.530     0.420   0.135        0.6770          0.2565          0.1415        0.2100      9
3      M   0.440     0.365   0.125        0.5160          0.2155          0.1140        0.1550     10
4      I   0.330     0.255   0.080        0.2050          0.0895          0.0395        0.0550      7
...   ..     ...       ...     ...           ...             ...             ...           ...    ...
4172   F   0.565     0.450   0.165        0.8870          0.3700          0.2390        0.2490     11
4173   M   0.590     0.440   0.135        0.9660          0.4390          0.2145        0.2605     10
4174   M   0.600     0.475   0.205        1.1760          0.5255          0.2875        0.3080      9
4175   F   0.625     0.485   0.150        1.0945          0.5310          0.2610        0.2960     10
4176   M   0.710     0.555   0.195        1.9485          0.9455          0.3765        0.4950     12
"""

label = data['Sex']
del data['Sex']

print(data.describe())
"""
            Length     Diameter       Height  Whole weight  Shucked weight  Viscera weight  Shell weight        Rings
count  4177.000000  4177.000000  4177.000000   4177.000000     4177.000000     4177.000000   4177.000000  4177.000000
mean      0.523992     0.407881     0.139516      0.828742        0.359367        0.180594      0.238831     9.933684
std       0.120093     0.099240     0.041827      0.490389        0.221963        0.109614      0.139203     3.224169
min       0.075000     0.055000     0.000000      0.002000        0.001000        0.000500      0.001500     1.000000
25%       0.450000     0.350000     0.115000      0.441500        0.186000        0.093500      0.130000     8.000000
50%       0.545000     0.425000     0.140000      0.799500        0.336000        0.171000      0.234000     9.000000
75%       0.615000     0.480000     0.165000      1.153000        0.502000        0.253000      0.329000    11.000000
max       0.815000     0.650000     1.130000      2.825500        1.488000        0.760000      1.005000    29.000000
"""

#print(data.info()) # <class 'pandas.core.frame.DataFrame'>



# Scaling하기

# data = (data - np.min(data)) / (np.max(data) - np.min(data))
# print(data)

# 위와 같이 코딩하지 않아도 MinMaxScaler를 사용하면 처리된다.





###### 1. MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
mMscaler = MinMaxScaler()

# mMscaler.fit(data)
# transfrom()
# mScaled_data = mMscaler.transform(data)
mScaled_data = mMscaler.fit_transform(data)
print(mScaled_data)
"""
[[0.51351351 0.5210084  0.0840708  ... 0.1323239  0.14798206 0.5       ]
 [0.37162162 0.35294118 0.07964602 ... 0.06319947 0.06826109 0.21428571]
 [0.61486486 0.61344538 0.11946903 ... 0.18564845 0.2077728  0.28571429]
 ...
 [0.70945946 0.70588235 0.18141593 ... 0.37788018 0.30543099 0.28571429]
 [0.74324324 0.72268908 0.13274336 ... 0.34298881 0.29347285 0.32142857]
 [0.85810811 0.84033613 0.17256637 ... 0.49506254 0.49177877 0.39285714]]
"""

# Numpy 배열을 Pandas에 있는 Data Frame 객체로 바꿔주기(보기 좋게 하기 위함)

mScaled_data = pd.DataFrame(mScaled_data, columns = data.columns)
print(mScaled_data)
"""
        Length  Diameter    Height  Whole weight  Shucked weight  Viscera weight  Shell weight     Rings
0     0.513514  0.521008  0.084071      0.181335        0.150303        0.132324      0.147982  0.500000
1     0.371622  0.352941  0.079646      0.079157        0.066241        0.063199      0.068261  0.214286
2     0.614865  0.613445  0.119469      0.239065        0.171822        0.185648      0.207773  0.285714
3     0.493243  0.521008  0.110619      0.182044        0.144250        0.149440      0.152965  0.321429
4     0.344595  0.336134  0.070796      0.071897        0.059516        0.051350      0.053313  0.214286
...        ...       ...       ...           ...             ...             ...           ...       ...
4172  0.662162  0.663866  0.146018      0.313441        0.248151        0.314022      0.246637  0.357143
4173  0.695946  0.647059  0.119469      0.341420        0.294553        0.281764      0.258097  0.321429
4174  0.709459  0.705882  0.181416      0.415796        0.352724        0.377880      0.305431  0.285714
4175  0.743243  0.722689  0.132743      0.386931        0.356422        0.342989      0.293473  0.321429
4176  0.858108  0.840336  0.172566      0.689393        0.635171        0.495063      0.491779  0.392857
"""





##### 2. Standard Scaler
from sklearn.preprocessing import StandardScaler
sdscaler = StandardScaler()

sdscaled_data = sdscaler.fit_transform(data)

# Numpy배열로 되어 있는 data를 Pandas Data Frame 객체로 바꾸기
sdscaled_data = pd.DataFrame(sdscaled_data, columns=data.columns)
print(sdscaled_data)
"""
        Length  Diameter    Height  Whole weight  Shucked weight  Viscera weight  Shell weight     Rings
0    -0.574558 -0.432149 -1.064424     -0.641898       -0.607685       -0.726212     -0.638217  1.571544
1    -1.448986 -1.439929 -1.183978     -1.230277       -1.170910       -1.205221     -1.212987 -0.910013
2     0.050033  0.122130 -0.107991     -0.309469       -0.463500       -0.356690     -0.207139 -0.289624
3    -0.699476 -0.432149 -0.347099     -0.637819       -0.648238       -0.607600     -0.602294  0.020571
4    -1.615544 -1.540707 -1.423087     -1.272086       -1.215968       -1.287337     -1.320757 -0.910013
...        ...       ...       ...           ...             ...             ...           ...       ...
4172  0.341509  0.424464  0.609334      0.118813        0.047908        0.532900      0.073062  0.330765
4173  0.549706  0.323686 -0.107991      0.279929        0.358808        0.309362      0.155685  0.020571
4174  0.632985  0.676409  1.565767      0.708212        0.748559        0.975413      0.496955 -0.289624
4175  0.841182  0.777187  0.250672      0.541998        0.773341        0.733627      0.410739  0.020571
4176  1.549052  1.482634  1.326659      2.283681        2.640993        1.787449      1.840481  0.640960
"""






##### 3. Sampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ros = RandomOverSampler()
rus = RandomUnderSampler()

# resampling

oversampled_data, oversampled_label = ros.fit_resample(data, label)
undersampled_data, undersampled_label = rus.fit_resample(data, label)

oversampled_data = pd.DataFrame(oversampled_data, columns=data.columns)
undersampled_data = pd.DataFrame(undersampled_data, columns=data.columns)

print('원본 데이터의 클래스 비율: \n{}'.format(pd.get_dummies(label).sum()))
print('Oversample 데이터의 클래스 비율: \n{}'.format(pd.get_dummies(oversampled_label).sum()))
print('Undersample 데이터의 클래스 비율: \n{}'.format(pd.get_dummies(undersampled_label).sum()))
"""
원본 데이터의 클래스 비율: 
F    1307
I    1342
M    1528
dtype: int64
Oversample 데이터의 클래스 비율:
F    1528
I    1528
M    1528
dtype: int64
Undersample 데이터의 클래스 비율:
F    1307
I    1307
M    1307
dtype: int64
"""






##### 4. classification 분류해주는 패키지. 샘플 데이터를 만들어준다.
# n_informative는 기본값 2로 값들 중에서 상관없이 튀는 데이터를 몇 개로 할 것인지 지정하는 것.
from sklearn.datasets import make_classification
data, label = make_classification(n_samples=1000,
                    n_features=2,
                    n_redundant=0,
                    n_informative=2,
                    n_repeated=0,
                    n_classes=3,
                    n_clusters_per_class=1,
                    weights=[0.05,0.15,0.8],
                    class_sep=0.8,
                    random_state=2019)

# 나오는 결과값: 첫번째==> 데이터, 두 번째==> 라벨

plt.Figure(figsize=(12, 6))
plt.scatter(data[:,0],data[:,1],c=label,alpha=0.3) # alpha 값=투명도




##### 5. SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE()

smoted_data, smoted_label = smote.fit_resample(data, label)

print('원본 데이터의 클래스 비율 \n{}'.format(pd.get_dummies(label).sum()))
print('\nSOMTE 결과 \n{}'.format(pd.get_dummies(smoted_label).sum()))
"""
원본 데이터의 클래스 비율 
0     53
1    154
2    793
dtype: int64

SOMTE 결과
0    793
1    793
2    793
dtype: int64
"""
fig = plt.Figure(figsize=(12,6))
plt.scatter(smoted_data[:,0],smoted_data[:,1],c=smoted_label,alpha=0.3)





##### 6. 차원의 축소(PCA)

from sklearn.datasets import load_digits
digits = load_digits()

print(digits.DESCR)

data = digits.data
label = digits.target

data.shape

label.shape

data[0].reshape(8,8)

label[0]

plt.imshow(data[0].reshape((8,8)))
print('Label: {}'.format(label[0]))

from sklearn.decomposition import PCA #주성분분석
pca = PCA(n_components=2)

new_data = pca.fit_transform(data)

print('원본 데이터의 차원 \n{}'.format(data.shape))
print('PCA를 거친 데이터의 차원 \n{}'.format(new_data.shape))

"""
Label: 0
원본 데이터의 차원 
(1797, 64)
PCA를 거친 데이터의 차원
(1797, 2)
"""

new_data[0] # data[0]을 2개의 차원으로 압축한 것

data[0] # 64차원

plt.scatter(new_data[:, 0], new_data[:, 1], c=label, alpha=0.4)
plt.legend()
plt.show()

data = pd.read_csv(abalone_path, header=None, names=abalone_columns)

label = data['Sex']

from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()

print(type(label)) # <class 'pandas.core.series.Series'>

print(label)
"""
0       M
1       M
2       F
3       M
4       I
       ..
4172    F
4173    M
4174    M
4175    F
4176    M
"""

# 위의 MFI 자료를 숫자로 바꿔주기
label_encoded_label = le.fit_transform(label)
label_encoded_label





##### 7. 원-핫 인코딩(One Hot Encoding). (sparse=False) 넣어야 배열 형태의 결과 나옴. 기본값은 True로 매트릭스 형태

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False) #True

one_hot_encoded = ohe.fit_transform(label.values.reshape((-1,1)))

print(one_hot_encoded)
"""
[[0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 ...
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]]
"""