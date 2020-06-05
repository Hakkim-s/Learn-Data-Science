import numpy as np

print('Finding LCM (Lowest Common Multiple)')
num1 = 4
num2 = 6
x = np.lcm(num1, num2)
print(x)

print('Finding LCM in Arrays')
arr = np.array([3, 6, 9])
x = np.lcm.reduce(arr)
print(x)

print('Find the LCM of all of an array where the array contains all integers from 1 to 10:')
arr = np.arange(1, 11)
x = np.lcm.reduce(arr)
print(x)

print('Finding GCD (Greatest Common Denominator)')
'''The GCD (Greatest Common Denominator), also known as HCF (Highest Common Factor) 
is the biggest number that is a common factor of both of the numbers.'''
num1 = 6
num2 = 9
x = np.gcd(num1, num2)
print(x)

print('Finding GCD in Arrays')
arr = np.array([20, 8, 32, 36, 16])
x = np.gcd.reduce(arr)
print(x)

print('Trigonometric Functions')
'''NumPy provides the ufuncs sin(), cos() and tan() that take values in
 radians and produce the corresponding sin, cos and tan values.'''
x = np.sin(np.pi/2)
print(x)

print('Find sine values for all of the values in arr:')
arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
x = np.sin(arr)
print(x)

print('Convert Degrees Into Radians')
arr = np.array([90, 180, 270, 360])
x = np.deg2rad(arr)
print(x)

arr = np.array([np.pi/2, np.pi, 1.5*np.pi, 2*np.pi])
x = np.rad2deg(arr)
print(x)

print('Finding Angles')
''' Finding angles from values of sine, cos, tan. E.g. sin, cos and tan inverse (arcsin, arccos, arctan).
NumPy provides ufuncs arcsin(), arccos() and arctan() that produce radian values
 for corresponding sin, cos and tan values given.'''
x = np.arcsin(1.0)
print(x)

print('Angles of Each Value in Arrays')
arr = np.array([1, -1, 0.1])
x = np.arcsin(arr)
print(x)

print('Hypotenues')
'''Finding hypotenues using pythagoras theorem in NumPy.
NumPy provides the hypot() function that takes the base and perpendicular
values and produces hypotenues based on pythagoras theorem.''' 
base = 3
perp = 4
x = np.hypot(base, perp)
print(x)

print('NumPy Hyperbolic Functions')
x = np.sinh(np.pi/2)
print(x)

print('Find cosh values for all of the values in arr:')
arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
x = np.cosh(arr)
print(x)    

print('Finding Angles')
'''Finding angles from values of hyperbolic sine, cos, tan. E.g. sinh, cosh and tanh inverse (arcsinh, arccosh, arctanh).
Numpy provides ufuncs arcsinh(), arccosh() and arctanh() that produce radian 
values for corresponding sinh, cosh and tanh values given.'''

x = np.arcsinh(1.0)
print(x)

print('Angles of Each Value in Arrays')

arr = np.array([0.1, 0.2, 0.5])
x = np.arctanh(arr)
print(x)

print('NumPy Set Operations')
'''Create Sets in NumPy
We can use NumPy's unique() method to find unique elements from any array.
E.g. create a set array, but remember that the set arrays should only be 1-D arrays.'''
arr = np.array([1, 1, 1, 2, 3, 4, 5, 5, 6, 7])
x = np.unique(arr)
print(x)    

print('To find the unique values of two arrays, use the union1d() method.')
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])
newarr = np.union1d(arr1, arr2)
print(newarr)

print('To find only the values that are present in both arrays, use the intersect1d() method.')
arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([3, 4, 5, 6])
newarr = np.intersect1d(arr1, arr2, assume_unique=True)
print(newarr)

print('Finding Difference')
'''Find the difference of the set1 from set2:'''
set1 = np.array([1, 2, 3, 4])
set2 = np.array([3, 4, 5, 6])
newarr = np.setdiff1d(set1, set2, assume_unique=True)
print(newarr)

print('Finding Symmetric Difference')
'''Find the symmetric difference of the set1 and set2:'''
set1 = np.array([1, 2, 3, 4])
set2 = np.array([3, 4, 5, 6])
newarr = np.setxor1d(set1, set2, assume_unique=True)
print(newarr)