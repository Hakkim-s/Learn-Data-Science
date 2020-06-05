#NumPy Searching Arrays
#You can search an array for a certain value, and return the indexes that get a match.

import numpy as np


arr = np.array([1, 2, 3, 4, 5, 4, 4,6,7])

x = np.where(arr == 4)
print(x)


#Find the indexes where the values are even:
y = np.where(arr%2 == 0)
print(y)
#example
x = np.where(arr%3 == 0)
print(x)
#Find the indexes where the values are odd:

x = np.where(arr%2 ==1)
print(x)


#Search Sorted
print("Search Sorted")
arr = np.array([6, 7, 8, 9])
x = np.searchsorted(arr, 7)
print(x)

x = np.searchsorted(arr, 7, side='right')
print(x)
    

arr = np.array([1, 3, 5, 7])

x = np.searchsorted(arr, [2, 4, 6])

print(x)