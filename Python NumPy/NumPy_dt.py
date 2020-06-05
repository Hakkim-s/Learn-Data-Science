
# NumPy Data types & copy , View

import numpy as np

arr = np.array([1, 2, 3 ,4])

print(arr.dtype)


arr_s = np.array(['apple', 'banana', 'cherry'])
print(arr_s.dtype)

arr = np.array([1, 2, 3 ,4], dtype='S')
print(arr_s)

print(arr_s.dtype)


#Converting Data_type
arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype('S')

print(newarr)
print(newarr.dtype)


arr = np.array([1,4,0,9,87,0,0,56,8,3,0])

newarr = arr.astype(bool)
print(newarr)

print('copy & View Examples Starts here')

print('copy doesn\'t gets affected')

arr = np.array([0,1,2,3,4])
new_arr = arr.copy()
arr[0] = 23
print(arr)
print(new_arr)

print('View gets affected')
arr = np.array([0,1,2,3,4])
new_arr = arr.view()
arr[0] = 23 
print(arr)
print(new_arr)
#Check if Array Owns it's Data
print(arr.base)
print(new_arr.base)