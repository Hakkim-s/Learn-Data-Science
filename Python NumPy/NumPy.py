# Array indexing, slicing

import numpy as np

arr = np.array([1,2,3])
print(np.__version__)
print(type(arr))

print(arr)
print(arr[1])



# 0-D
arr = np.array(42)

print(arr)
# 1-D
arr = np.array([1,2,3], ndmin=5)
print(arr)
arr = np.array([0,1,2,3,4,5,6,7])

#slicing_examples
print('Slicing Examples')
print(arr[0:4])
print(arr[0:])
print(arr[:6])
print(arr[-4:-1])
print(arr[:6:3])
#Return every other element from the entire array:
print(arr[::2])
#2-D

arr = np.array([[1,2,3], [4,5,6]])
print(arr[1,1])

arr =  np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(arr[0, 1:4])
print(arr[0:2, 1:4])

#From both elements, return index 4:
print(arr[0:2, 4])


#3-D

arr = np.array([[[1,2,3], [4,5,6]],[[7,8,9], [10,11,12]]])
print(arr.ndim)

#Access the third element of the second array of the first array:
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr[1,1,2])

#Print the last element from the 2nd dim:

arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

print(arr[1,1,-0])
