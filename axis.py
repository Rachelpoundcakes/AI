import numpy as np
a1 = np.array([1, 2, 3, 4, 5])

print(type(a1))
# <class 'numpy.ndarray'>

print(a1)
#[1, 2, 3, 4, 5] => axis0(행)만 존재

a2 = np.arange(8).reshape(2, 4)
print(a2)
"""
[[0 1 2 3]
 [4 5 6 7]]
 => axis0 2행, axis1 열
"""

a3 = np.arange(12).reshape(2, 3, 2)                          
print(a3)
"""
[[[ 0  1]
  [ 2  3]
  [ 4  5]]

 [[ 6  7]
  [ 8  9]
  [10 11]]]
  => 행(row) 2개, column 3개, depth 2개
  * 축의 방향성
  axis0  [0 1] -> [6 7]
  axis1 [0 1] -> [2 3]
  axis2 0 -> 1 
"""
