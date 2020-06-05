import numpy as np
#What are ufuncs?
print('ufuncs stands for "Universal Functions" and they are NumPy functions that operates on the ndarray object.')
'''Why use ufuncs?
ufuncs are used to implement vectorization in NumPy which is way faster than iterating over elements.

They also provide broadcasting and additional methods like reduce, accumulate etc. that are very helpful for computation.

ufuncs also take additional arguments, like:

where boolean array or condition defining where the operations should take place.

dtype defining the return type of elements.

out output array where the return value should be copied.'''


x = [1, 2, 3, 4]
y = [4, 5, 6, 7]
z = []
print('Without ufunc, we can use Python\'s built-in zip() method:')
for i, j in zip(x, y):
  z.append(i + j)
print(z)

print('With ufunc, we can use the add() function:')
z = np.add(x, y)
print(z)

'''How To Create Your Own ufunc
To create you own ufunc, you have to define a function, like you do with  
normal functions in Python, then you add it to your NumPy ufunc library with the frompyfunc() method.

The frompyfunc() method takes the following arguments:

function - the name of the function.
inputs - the number of input arguments (arrays).
outputs - the number of output arrays.'''


def myadd(x, y):
  return x+y

myadd = np.frompyfunc(myadd,2,1)
print(myadd([1, 2, 3, 4], [5, 6, 7, 8]))

print('Check if a Function is a ufunc')
print(type(np.add))
'''If it is not a ufunc, it will return another type, like this built-in 
NumPy function for joining two or more arrays:'''
print(type(np.concatenate))
print(type(np.append))

if type(np.add) == np.ufunc:
  print('add is ufunc')
else:
  print('add is not ufunc')
  
print('Simple Arithmetic')
#Addition
arr1 = np.array([10, 11, 12, 13, 14, 15])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.add(arr1, arr2)
print(newarr)

#subtraction
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([20, 21, 22, 23, 24, 25])

newarr = np.subtract(arr1, arr2)
print(newarr)  

#Multiplication
print(np.multiply(arr1, arr2))

#divide
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 5, 10, 8, 2, 33])

newarr = np.divide(arr1, arr2)
print(newarr)

#Power
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 5, 6, 8, 2, 33])

newarr = np.power(arr1, arr2)
print(newarr)

#Remainder/mod
arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 2, 33])

newarr = np.mod(arr1, arr2)

print(newarr)
print(np.remainder(arr1, arr2))

#Return the quotient and mod:

arr1 = np.array([10, 20, 30, 40, 50, 60])
arr2 = np.array([3, 7, 9, 8, 2, 33])
# 1st array Quotient and 2nd array Remainder
newarr = np.divmod(arr1, arr2)
print(newarr)

#Absolute Values

arr = np.array([-1, -2332, 1, 2, 3, -4])

newarr = np.absolute(arr)
print(newarr)