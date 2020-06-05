#Random
#Computers work on programs, and programs are definitive set of instructions.
#So it means there must be some algorithm to generate a random number as well.

from numpy import random
print('Generate Random Number')
x = random.randint(1000)
print(x)
print('Generate Random Float')
x = random.rand()
print(x)
print('Generate Random Array')
x=random.randint(100, size=(58))
print(x)
#2-D
x = random.randint(100, size=(3, 5))
print(x)
x = random.rand(5)
print(x)
x = random.rand(5,5)
print(x)
print('Generate Random Number From Array')
x = random.choice([3, 5, 7, 9])
y = random.choice([3, 5, 7, 9], size=(3, 5))
print(x)
print(y)    

print('Random Distribution')
#The sum of all probability numbers should be 1.
x = random.choice([3, 5, 7, 9, 6], p=[0.1, 0.1, 0.3, 0.5 ,0], size=(100))
xy= random.choice([3, 5, 7, 9, 6], p=[0.1, 0.1, 0.3, 0.5 ,0], size=(2,4))
print(x)
print(y)    
print("Random Permutations of Elements")
print('Shuffling Arrays')
from numpy import random
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

random.shuffle(arr)

print(arr)
print('Generating Permutation of Arrays')

arr = np.array([1, 2, 3, 4, 5])

print(random.permutation(arr))
