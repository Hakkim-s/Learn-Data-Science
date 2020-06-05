import numpy as np

arr = np.array([[1, 2, 3, 4,4], [5, 6, 7, 8,44],[1, 2, 3, 4,4], [5, 6, 7, 8,32]])

print(arr.shape)


arr = np.array([1, 2, 3, 4], ndmin=3)
print(arr)
print('shape of array :', arr.shape)



arr = np.array([1, 2, 3, 4,5], ndmin=5)
print(arr)
print('shape of array :', arr.shape)


#NumPy Array Reshaping

print('1-D to 2-D')
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

newarr = arr.reshape(4, 3)
print(newarr)

print('1-D to 3-D')

newarr = arr.reshape(2,3,2)

print(newarr)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])

#Check if the returned array is a copy or a view:
print(arr.reshape(2,2,2).base)

#Pass -1 as the value, and NumPy will calculate this number for you.
newarr = arr.reshape(2, 2, -1)

print(newarr)

#Flattening array means converting a multidimensional array into a 1D array.
arr = np.array([[1, 2, 3], [4, 5, 6]])

newarr = arr.reshape(-1)

print(newarr)
