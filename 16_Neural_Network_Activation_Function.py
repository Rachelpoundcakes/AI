# 활성화 함수(Activation Function)
import numpy as np
import matplotlib.pyplot as plt

# Step Function (계단 함수): 불연속 함수이기 때문에 미분이 불가능, 데이터 손실 가능성.
def step_function(x):
  if x > 0:
    return 1
  else:
    return 0

def step_function_for_numpy(x):
  y = x > 0
  value  = y.astype(np.int)

  return value

print(step_function(-3))
print(step_function(5))

"""
0
1
"""

# 시그모이드 함수(Sigmoid): 이진분류에 사용
def sigmoid(x):
  value = 1 / (1 + np.exp(-x))
  return value

print(sigmoid(3))
print(sigmoid(-3))
"""
0.9525741268224334
0.04742587317756678
"""
plt.grid()
x = np.arange(-5,5,0.01)
y1 = sigmoid(x)
y2 = step_function_for_numpy(x)

plt.plot(x, y1,'r-')
plt.plot(x, y2, 'b--')
plt.show()

# ReLu(x): 입력이 0을 넘으면 그 입력을 그대로 출력하고, 0 이하면 0을 출력하는 함수
def ReLu(x):
  if x > 0:
    return x
  else:
    return 0

# Identity Function (항등 함수): 회귀에서 사용
def identify_function(x):
  return x

# Softmax(a): 다중 분류에서 사용. 출력값의 총합은 1
def Softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

a = np.array([0.3,0.2,3.0,-1.2])
print(Softmax(a))
print(np.sum(Softmax(a)))

"""
[0.0587969  0.05320164 0.8748821  0.01311936]
1.0
"""