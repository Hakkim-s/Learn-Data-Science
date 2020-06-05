print('Poisson Distribution')
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

x = random.poisson(lam=2, size=10)
print(x)
sns.distplot(random.poisson(lam=2, size=1000), kde=False)
plt.show()
'''Difference Between Normal and Poisson Distribution
Normal distribution is continous whereas poisson is discrete.

But we can see that similar to binomial for a large enough poisson distribution
 it will become similar to normal distribution with certain std dev and mean.'''
 
sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson')
plt.show()

'''Difference Between Poisson and Binomial Distribution
The difference is very subtle it is that, 
binomial distribution is for discrete trials, whereas poisson distribution is for continuous trials.

But for very large n and near-zero p 
binomial distribution is near identical to poisson distribution such that n * p is nearly equal to lam.'''

sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')
plt.show()

print('Uniform Distribution')
'''Used to describe probability where every event has equal chances of occuring.'''
x = random.uniform(size=(2, 3))
print(x)
sns.distplot(random.uniform(size=1000), hist=False)
plt.show()

print('Logistic Distribution')
'''Logistic Distribution is used to describe growth.
Used extensively in machine learning in logistic regression, neural networks etc.'''
x = random.logistic(loc=1, scale=2, size=(2, 3))
print(x)
sns.distplot(random.logistic(size=1000), hist=False)
plt.show()

'''Difference Between Logistic and Normal Distribution
Both distributions are near identical, but logistic distribution has more area under the tails
.ie. It representage more possibility of occurence of an events further away from mean.

For higher value of scale (standard deviation) the normal and
 logistic distributions are near identical apart from the peak.'''
sns.distplot(random.normal(scale=2, size=1000), hist=False, label='normal')
sns.distplot(random.logistic(size=1000), hist=False, label='logistic')

plt.show()

print('Multinomial Distribution')
'''Multinomial distribution is a generalization of binomial distribution.
It describes outcomes of multi-nomial scenarios unlike binomial where scenarios must be only one of two.
 e.g. Blood type of a population, dice roll outcome.'''
x = random.multinomial(n=6, pvals=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
print(x)

print('Exponential Distribution')
'''Exponential distribution is used for describing time till next event e.g. failure/success etc.'''
x = random.exponential(scale=2, size=(2, 3))
print(x)
sns.distplot(random.exponential(size=1000), hist=False)
plt.show()
'''Relation Between Poisson and Exponential Distribution
Poisson distribution deals with number of occurences of an event in a time period whereas
 exponential distribution deals with the time between these events.'''
 
print('Chi Square Distribution')
'''Chi Square distribution is used as a basis to verify the hypothesis.'''
x = random.chisquare(df=2, size=(2, 3))
print(x)

sns.distplot(random.chisquare(df=1, size=1000), hist=False)
plt.show()

print('Rayleigh Distribution')
'''Rayleigh distribution is used in signal processing.'''
x = random.rayleigh(scale=2, size=(2, 3))
print(x)
sns.distplot(random.rayleigh(size=1000), hist=False)
plt.show()
'''Similarity Between Rayleigh and Chi Square Distribution
At unit stddev the and 2 degrees of freedom rayleigh and
chi square represent the same distributions.'''
print('Pareto Distribution')
'''A distribution following Pareto's law 
i.e. 80-20 distribution (20% factors cause 80% outcome).'''
sns.distplot(random.pareto(a=2, size=1000), kde=False)

plt.show()
print('Zipf Distribution')
'''Zipf's Law: In a collection the nth common term is 1/n times of the most common term.
 E.g. 5th common word in english has occurs nearly 1/5th times as of the most used word.'''
x = random.zipf(a=2, size=(2, 3))
print(x)
'''Sample 1000 points but plotting only ones with value < 10 for more meaningful chart.'''

x = random.zipf(a=2, size=1000)
sns.distplot(x[x<10], kde=False)
plt.show()