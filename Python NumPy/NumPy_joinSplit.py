#NumPy Joining Array
print('NumPy Joining Array Starts here')
import numpy as np

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

#We pass a sequence of arrays that we want to join to the concatenate() function,
# along with the axis. If axis is not explicitly passed, it is taken as 0.
print('Concatenating')
arr = np.concatenate((arr1,arr2))

print(arr)

arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)

print(arr)


#Joining Arrays Using Stack Functions
print('Stacking 1-D')

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)
print(arr)
#Satcking alone rows
arr = np.hstack((arr1, arr2))
print(arr)

#Satcking alone columns
arr = np.vstack((arr1, arr2))
print(arr)

#Stacking Along Height (depth)
arr = np.dstack((arr1, arr2))
print(arr)



arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])
print('Stacking 2-D')

arr = np.stack((arr1, arr2), axis=1)
print(arr)
#Satcking alone rows

arr = np.hstack((arr1, arr2))
print(arr)

#Satcking alone columns
arr = np.vstack((arr1, arr2))
print(arr)

#Stacking Along Height (depth)
arr = np.dstack((arr1, arr2))
print(arr)



#Splitting NumPy Arrays
print('Splitting NumPy Arrays Starts here')

arr = np.array([1,2,3,4,5,6,7,8,9,10])
newarr = np.array_split(arr, 3)

print(newarr)
#Split Into Arrays
print(newarr[0])
print(newarr[1])
print(newarr[2])
newarr2 = np.array_split(arr, 4)
print(newarr2)
#Split Into Arrays
print(newarr2[3])

#2-D
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13,14]])

newarr = np.array_split(arr, 3)

print(newarr)
print(newarr[0])
print(newarr[1])
print(newarr[2])

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3)
print(newarr)

#The example below also returns three 2-D arrays, 
#but they are split along the row (axis=1).
newarr = np.array_split(arr, 3, axis=1)
print(newarr)
newarr = np.hsplit(arr, 3)
print(newarr)
newarr = np.vsplit(arr, 3)
print(newarr)

