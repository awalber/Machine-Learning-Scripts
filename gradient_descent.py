# This code was written by Aaron Walber to display an understanding of machine learning
# topics taught in Andrew Ng's online machine learning course found on coursera. The
# course is taught in Octave, and I wanted to translate what I learned
# to python for my own benefit. The data used in this 
# code is from the first exercise of that course.
from numpy import power, sum, zeros, matrix, shape, arange
from pylab import xlabel, ylabel, title, plot, show
from pandas import read_csv
from matplotlib import pyplot
import os

def compute_cost(X,y,theta,L):
    theta_sqrd = power(theta[1:],2)
    j = power(((X*theta.T)-y),2)
    J = sum(j)/(2*m)+(L/(2*m))*sum(theta_sqrd)
    return J

def gradient_descent(X,y,theta,alpha,iters):
    J_vector = zeros((iters,1))
    for i in range(iters):
        delta = (X*theta.T) - y
        theta = theta - (alpha/m)*delta.T*X
        J_vector[i] = compute_cost(X,y,theta,0)
    return theta,J_vector

# This data is taken from Andrew Ng's machine learning course found on coursera.
# However, this code should work for any data set that can be split into a matrix of
# input features and the resulting output.
data = read_csv("D:/Python/Python/Custom_Projects/Machine_Learning/ex1data1.txt")
# The next line normalizes the data by transforming the set into z-scores
data = (data - data.mean()) / data.std()
# Add a column of ones to the data set. This is standard when running gradient descent
data.insert(0, 'Ones', 1)
# Choose the desired values for x and y
X = data.iloc[:,:-1]
y = data.iloc[:,-1:]
# Translate the values into a matrix format from pandas formatting
X = matrix(X.values)
y = matrix(y.values)
# Initiate some variables used in the function
m = len(y)
iters = 700
alpha = .005
# Set original theta to a vector of zeros with the same shape as X
theta = zeros(shape(X[1,:]))
# Call our gradient descent function and plot the cost versus iterations.
# This verifies that the amount of iterations and the value for alpha are
# properly chosen so the data isn't over/underfitted.
Theta, cost = gradient_descent(X,y,theta,alpha,iters)
plot(arange(iters), cost, 'r')
xlabel('Iterations')
ylabel(r'J($\theta$)')
title('Cost function vs. Iterations')
show()
# Here we plot the original data found using the
# optimized value of theta that minimizes the cost function
plot(X[:,1], X@Theta.T, 'r',linewidth=.5)
pyplot.scatter([X[:,1]],[y],s=16,marker='.')
show()