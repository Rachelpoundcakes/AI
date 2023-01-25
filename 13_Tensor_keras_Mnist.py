import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


# 0차원 텐서
x = np.array(3) # 1개짜리 스칼라 값, 방향성이 없다.
print(x) # 3
print(x.shape) # ()
print(np.ndim(x)) # 0 ==> 차원이 없다.

# 벡터(1차원 텐서)
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])

# 벡터(1차원 텐서)의 연산: 덧셈
c = a + b
print(c) # [ 6  8 10 12]
print(c.shape) # (4,)
print(np.ndim(c)) # 1 ==> 1차원이다.

# 벡터(1차원 텐서)의 연산: 곱셈
c = a * b
print(c) # [ 5 12 21 32]
print(c.shape) # (4,)
print(np.ndim(c)) # 1 ==> 1차원이다.

#스칼라와 벡터의 곱. 일괄적용에 쓰인다. 예) 물품 항목 * 1.04(물가 0.4% 상승 반영)
a = np.array(10) # 스칼라(0차원 텐서)
b = np.array([1,2,3]) # 벡터(1차원 텐서)
c = a * b

print(c) # [10 20 30]

# 전치행렬(행과 열을 바꾼 배열의 형태)
# 2차원 텐서

A = np.array([[1,3,4], [4,5,6]])
print('A\n', A)
print('A.shape\n', A.shape)
print('--------------------')

A_ = A.T
print('A_\n', A_)
print('A_.shape\n', A_.shape)
print('--------------------')

"""
A
 [[1 3 4]
 [4 5 6]]
A.shape
 (2, 3)
--------------------
A_
 [[1 4]
 [3 5]
 [4 6]]
A_.shape
 (3, 2)
--------------------
"""


#순서대로 --> (학습용 데이터, 라벨),(테스트용 데이터, 라벨)
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.ndim) # 3

print(train_images.shape) # (60000, 28, 28) ==> 총 60000만장, 사이즈 28픽셀 이미지로 구성되어 있다.

print(train_images.dtype) # uint8 ==> 음의 정수가 없는, 양의 정수만 있는 8비트 타입의 정수형 ==> 2^8, 255까지 색상 표현이 가능

#제일 첫 번째 장 가져오기 #color map
train_images[0]
temp_image = train_images[0]
plt.imshow(temp_image,cmap='gray')
plt.show()

print(train_labels[0]) # 5
print(train_labels[3]) # 1