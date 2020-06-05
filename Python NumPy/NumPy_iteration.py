#NumPy Array Iterating
import numpy as np

arr = np.array([[1, 2, 3], [4, 5, 6]])

for x in arr:
  print(x)
  
#Iterate on each scalar element of the 2-D array:

for x in arr:
    for y in x:
        print(y)
        
        
        
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  print(x)
  
for x in arr:
    for y in x:
        for z in y:
            print(z)
            
#The function nditer() is a helping function that can be used from very basic to
#very advanced iterations. It solves some basic issues which we face in iteration,
# lets go through it with examples.
print('nditer() Example')            
for x in np.nditer(arr):
    print(x)            
            
print('Iterating Array With Different Data Types')    

for x in np.nditer(arr ,flags=['buffered'],op_dtypes =['S']):
    print(x)            
            
#Iterating With Different Step Size
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

for x in np.nditer(arr[:, ::2]):
  print(x)    
  
#Enumerated Iteration Using ndenumerate()
#Sometimes we require corresponding index of the element while iterating,
#the ndenumerate() method can be used for those usecases.  
  
for idx,x in np.ndenumerate(arr):
    print(idx,x)  
    
    
    
arr = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]])
for idx, x in np.ndenumerate(arr):
  print(idx, x)     