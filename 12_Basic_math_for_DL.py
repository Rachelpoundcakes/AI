import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def linear_function(x):
  a = 0.8 # 기울기
  b = 2 # y절편은 고정값

  return a * x + b

print(linear_function(5))
# 6.0

np.arange(-5, 5, 0.1) #-5부터 5까지 0.1씩 배열 만들기

# X축의 값으로 저장
x = np.arange(-5, 5, 0.1)
y = linear_function(x)
print(y)
"""
[-2.00000000e+00 -1.92000000e+00 -1.84000000e+00 -1.76000000e+00
 -1.68000000e+00 -1.60000000e+00 -1.52000000e+00 -1.44000000e+00
...
5.68000000e+00  5.76000000e+00  5.84000000e+00  5.92000000e+00]
"""

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Function')
plt.show()

# 이차함수 y = ax^2 + bx + c

def quadratic_function(x):
  a = 1
  b = -1
  c = -2

  return a*x**2 + b*x + c

print(quadratic_function(2))
# 0

y = quadratic_function(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Function')
plt.show()

# 삼차함수(다항함수) y=ax3 + bx2 + cx + d

def cubic_function(x):
  a = 4
  b = 0
  c = -1
  d = -8

  return a*x**3 + b*x**2 + c*x+d

y = cubic_function(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Function')

# 최소값, 최대값
# 하강법

def my_func(x):
  a = 1
  b = -3
  c = 10

  return a*x**2 + b*x + c # 이차함수의 수식

x = np.arange(-10, 10, 0.1) # -10부터 10까지 0.1씩
y = my_func(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('My Function')
plt.show()

# 점 찍기
plt.scatter(1.5, my_func(1.5))

# 지수 함수 y = ax
def exponential_function(x):
  a = 4
  return a**x

print(exponential_function(4))
print(exponential_function(0))
"""
256
1
"""

x = np.arange(-3, 2, 0.1)
y = exponential_function(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-4, 3)
plt.ylim(-1, 15)
plt.title('exponential_function')
plt.show()

