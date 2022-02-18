import numpy as np


x = np.arange(9.0).reshape((3,3))
y = np.arange(9.0).reshape((3,3))
# x = x.reshape(5,1)
# x = x.reshape(1,5)
# y = y.reshape(5,1)
print('X: ', x)
print('Y: ', y)


z = x*2
print('Z: ', z)
print(x+y)