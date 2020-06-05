from math import log
import numpy as np

print('Rounding Decimals')

'''There are primarily five ways of rounding off decimals in NumPy:

truncation
fix
rounding
floor
ceil    '''

print('Truncation')

arr = np.trunc([-3.1666, 3.6667])

print(arr)
print(np.fix([-3.1666, 3.6667]))

print('Rounding')
arr = np.around(3.1666, 2)
print(arr)

print('floor')
#The floor() function rounds off decimal to nearest lower integer.
arr = np.floor([-3.166612212, 3.12126667])

print(arr)

print('ceil')
#The ceil() function rounds off decimal to nearest upper integer.
arr = np.ceil([-3.1666, 3.6667])
print(arr)


#NumPy Logs
print('Log at Base 2')
arr = np.arange(1, 10)
print(np.log2(arr))

print('Log at Base 10')

arr = np.arange(1, 10)
print(np.log10(arr))

print('Natural Log, or Log at Base e')

arr = np.arange(1, 10)

print(np.log(arr))

print('Log at Any Base')
nplog = np.frompyfunc(log, 2, 1)

print(nplog(100, 15))

print('Summations')
#Sum the the values in arr1 and the values in arr2:
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])
newarr = np.sum([arr1, arr2])
print(newarr)

print('Summation Over an Axis')
#Summation Over an Axis
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])
newarr = np.sum([arr1, arr2], axis=1)
print(newarr)

print('Cummulative Sum')

'''Cummulative sum means partially adding the elements in array.
E.g. The partial sum of [1, 2, 3, 4] would be
 [1, 1+2, 1+2+3, 1+2+3+4] = [1, 3, 6, 10].'''
 
arr = np.array([1, 2, 3])
newarr = np.cumsum(arr)
print(newarr)

print('NumPy Products')

arr = np.array([1, 2, 3, 4])
x = np.prod(arr)
print(x)

#Find the product of the elements of two arrays:
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
x = np.prod([arr1, arr2])
print(x)

#Product Over an Axis
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])
newarr = np.prod([arr1, arr2], axis=1)
print(newarr)

print('Cummulative Product')
'''Cummulative product means taking the product partially.
E.g. The partial product of [1, 2, 3, 4] is
 [1, 1*2, 1*2*3, 1*2*3*4] = [1, 2, 6, 24]'''
arr = np.array([5, 6, 7, 8])
newarr = np.cumprod(arr)
print(newarr)

print('NumPy Differences')
'''A discrete difference means subtracting two successive elements.
E.g. for [1, 2, 3, 4], the discrete difference would be
 [2-1, 3-2, 4-3] = [1, 1, 1]'''
 
arr = np.array([10, 15, 25, 5])
newarr = np.diff(arr)
print(newarr)

'''E.g. for [1, 2, 3, 4], the discrete difference with n = 2 would be [2-1, 3-2, 4-3] = [1, 1, 1] ,
then, since n=2, we will do it once more, with the new result: [1-1, 1-1] = [0, 0]'''

arr = np.array([10, 15, 25, 5])
newarr = np.diff(arr, n=2)
print(newarr)