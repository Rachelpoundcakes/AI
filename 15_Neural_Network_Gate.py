# AND 게이트 가중치 만들기
import numpy as np
import matplotlib.pyplot as plt

def AND(a, b):
  input = np.array([a,b])

  # 가중치 설정
  weights = np.array([0.4, 0.4])
  bias = -0.6

  # 출력값
  value = np.sum(input * weights) + bias
  
  # ==> 이렇게 하면 뉴럴 한 개의 프로그램을 짠 것

  # 반환값
  if value <= 0:
    return 0
  else:
    return 1

print(AND(0,0))
print(AND(0,1))
print(AND(1,0))
print(AND(1,1))

"""
0
0
0
1
"""

# AND 게이트 시각화
x1 = np.arange(-2, 2, 0.01) # input 값
x2 = np.arange(-2, 2, 0.01) # output 값
bias = -0.6

y = (-0.4 * x1 - bias) / 0.4

plt.plot(x1, y, 'r--')
plt.scatter(0,0, color='orange', marker='o',s=150)
plt.scatter(0,1, color='orange', marker='o',s=150)
plt.scatter(1,0, color='orange', marker='o',s=150)
plt.scatter(1,1, color='black', marker='^',s=150)
plt.xlim(-0.5,1.5)
plt.ylim(-0.5,1.5)
plt.grid()
plt.show()

# OR 게이트 가중치 만들기
def OR(a, b):
  input = np.array([a,b])

  #가중치 설정
  weights = np.array([0.4,0.4])
  bias = -0.3

  #출력값
  value = np.sum(input * weights) + bias

  #반환값
  if value <= 0:
    return 0
  else:
    return 1

print(OR(0,0))
print(OR(0,1))
print(OR(1,0))
print(OR(1,1))

"""
0
1
1
1
"""

# NAND 게이트 가중치 만들기
def NAND(a, b):
  input = np.array([a,b])

  #가중치 설정
  weights = np.array([-0.6,-0.6])
  bias = 0.7

  #출력값
  value = np.sum(input * weights) + bias

  #반환값
  if value <= 0:
    return 0
  else:
    return 1

print(NAND(0,0))
print(NAND(0,1))
print(NAND(1,0))
print(NAND(1,1))

"""
1
1
1
0
"""

# XOR 가중치 만들기
def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  y = AND(s1, s2)

  return y

print(XOR(0,0))
print(XOR(0,1))
print(XOR(1,0))
print(XOR(1,1))

"""
0
1
1
0
"""
