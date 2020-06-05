#Sorting Arrays

import numpy as np

arr = np.array([3, 2, 0, 1])
print(np.sort(arr))


arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(arr))


arr = np.array([True, False, True])
print(np.sort(arr))



#2_D
arr = np.array([[3, 2, 4], [5, 0, 1]])

print(np.sort(arr))

#Filtering Arrays
print('Filtering Arrays Starts Here')
arr = np.array([41, 42, 43, 44])

x = arr[[True, False, True, False]]

print(x)

filter_arr = []

for ele in arr:
    if ele > 42:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
        
newarr = arr[filter_arr]
print(arr)
print(newarr)
#even num
print('even_numbers')
arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = []

for ele in arr:
    if ele % 2==0:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
        
newarr = arr[filter_arr]
print(arr)
print(newarr)

print('Creating Filter Directly From Array')
arr = np.array([41, 42, 43, 44])
filter_arr = arr > 42
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)


arr = np.array([1, 2, 3, 4, 5, 6, 7])
filter_arr = arr%2 ==0
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)



