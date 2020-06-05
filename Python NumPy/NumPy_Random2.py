print('Visualize Distributions With Seaborn')

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random

sns.distplot([0,1,2,3,4,5])
plt.show()

sns.distplot([0,1,2,3,4,5], hist=False)
plt.show()

print('Normal (Gaussian) Distribution')

x = random.normal(size=(2, 3))  
print(x)

'''Use the random.normal() method to get a Normal Data Distribution.
It has three parameters:
loc - (Mean) where the peak of the bell exists.
scale - (Standard Deviation) how flat the graph distribution should be.
size - The shape of the returned array.'''
x = random.normal(loc=1, scale=2, size=(2, 3))
print(x)

print('Visualization of Normal Distribution')
sns.distplot(random.normal(size=1000), hist=False)
plt.show()


print('Binomial Distribution')
'''Binomial Distribution is a Discrete Distribution.
It describes the outcome of binary scenarios, e.g. toss of a coin, it will either be head or tails.
It has three parameters:
n - number of trials.
p - probability of occurence of each trial (e.g. for toss of a coin 0.5 each).
size - The shape of the returned array.
'''
x = random.binomial(n=10, p=0.5, size=10)
print(x)

print('Visualization of Binomial Distribution')
sns.distplot(random.binomial(n=10, p=0.5, size=1000), hist=True, kde=False)
plt.show()

print('Difference Between Normal and Binomial Distribution')
sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')
plt.show()
